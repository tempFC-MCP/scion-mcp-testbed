// SCION-native proposal workload. The build uses the pinned upstream HTTP/3
// example's scion.go helper, with its original Apache-2.0 notice preserved.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/quic-go/quic-go"
	"github.com/quic-go/quic-go/http3"
	"github.com/scionproto/scion/pkg/snet"
	"github.com/scionproto/scion/pkg/snet/addrutil"
	"github.com/scionproto/scion/private/app/path"
	"golang.org/x/sys/unix"
)

// Both peers must permit idle gaps between a low-quota proposer's turns.
// QUIC negotiates the smaller peer timeout.
const connectionIdleTimeout = 5 * time.Minute

func mono() int64 {
	var t unix.Timespec
	if err := unix.ClockGettime(unix.CLOCK_MONOTONIC, &t); err != nil {
		panic(err)
	}
	return t.Nano()
}
func waitUntil(ns int64) {
	if d := ns - mono(); d > 0 {
		time.Sleep(time.Duration(d))
	}
}

type logFile struct {
	mu sync.Mutex
	f  *os.File
}

func newLog(p string) *logFile {
	f, e := os.OpenFile(p, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if e != nil {
		panic(e)
	}
	return &logFile{f: f}
}
func (l *logFile) put(v any) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if e := json.NewEncoder(l.f).Encode(v); e != nil {
		panic(e)
	}
}
func save(p string, v any) error {
	b, e := json.Marshal(v)
	if e != nil {
		return e
	}
	return os.WriteFile(p, b, 0644)
}

func serve(daemonAddr, listen, logPath string) error {
	ctx := context.Background()
	dc, topo, e := dialDaemon(ctx, daemonAddr)
	if e != nil {
		return e
	}
	defer dc.Close()
	address, e := net.ResolveUDPAddr("udp", listen)
	if e != nil {
		return e
	}
	network := &snet.SCIONNetwork{Topology: topo, SCMPHandler: ignoreSCMP{}}
	conn, e := network.Listen(ctx, "udp", address)
	if e != nil {
		return e
	}
	defer conn.Close()
	cert, e := selfSignedCert()
	if e != nil {
		return e
	}
	log := newLog(logPath)
	defer log.f.Close()
	mux := http.NewServeMux()
	mux.HandleFunc("/ready", func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "ready") })
	mux.HandleFunc("/warmup", func(w http.ResponseWriter, r *http.Request) {
		n, err := io.Copy(io.Discard, http.MaxBytesReader(w, r.Body, 4*1024*1024))
		if err != nil || n != r.ContentLength {
			http.Error(w, "incomplete warmup", 400)
			return
		}
		fmt.Fprint(w, "warm")
	})
	mux.HandleFunc("/proposal", func(w http.ResponseWriter, r *http.Request) {
		started := mono()
		first := int64(0)
		total := int64(0)
		h := sha256.New()
		buf := make([]byte, 32*1024)
		var readErr error
		body := http.MaxBytesReader(w, r.Body, 4*1024*1024)
		for {
			n, err := body.Read(buf)
			if n > 0 {
				if first == 0 {
					first = mono()
				}
				total += int64(n)
				_, _ = h.Write(buf[:n])
			}
			if err != nil {
				if err != io.EOF {
					readErr = err
				}
				break
			}
		}
		complete := mono()
		expected, _ := strconv.ParseInt(r.Header.Get("X-Bytes"), 10, 64)
		hash := hex.EncodeToString(h.Sum(nil))
		valid := readErr == nil && total == expected && hash == r.Header.Get("X-Hash")
		event := map[string]any{"kind": "receipt", "run": r.Header.Get("X-Run"), "round": r.Header.Get("X-Round"),
			"validator": r.Header.Get("X-Validator"), "attempt": r.Header.Get("X-Attempt"), "path": r.Header.Get("X-Path"),
			"receiver": topo.LocalIA.String(), "handler_ns": started, "first_byte_ns": first, "complete_ns": complete,
			"bytes": total, "hash": hash, "verified": valid}
		if readErr != nil {
			event["error"] = readErr.Error()
		}
		log.put(event)
		if !valid {
			http.Error(w, "payload verification failed", 400)
			return
		}
		fmt.Fprint(w, "verified")
	})
	qc := quicConfig()
	qc.MaxIdleTimeout = connectionIdleTimeout
	srv := &http3.Server{Handler: mux, TLSConfig: &tls.Config{Certificates: []tls.Certificate{cert}, NextProtos: []string{alpn}}, QUICConfig: qc}
	return srv.Serve(conn)
}

type route struct {
	Remote      string `json:"remote"`
	Sequence    string `json:"sequence"`
	Fingerprint string `json:"fingerprint"`
	Core        string `json:"core"`
}
type job struct {
	Round     int       `json:"round"`
	Validator string    `json:"validator"`
	Targets   [][]route `json:"targets"`
}
type config struct {
	Run              string `json:"run"`
	Daemon           string `json:"daemon"`
	Jobs             []job  `json:"jobs"`
	Rounds           int    `json:"rounds"`
	IntervalNS       int64  `json:"interval_ns"`
	DeadlineNS       int64  `json:"deadline_ns"`
	Payload          int    `json:"payload_bytes"`
	StartFile        string `json:"start_file"`
	Output           string `json:"output"`
	Ready            string `json:"ready"`
	FailedCore       string `json:"failed_core"`
	Routing          string `json:"routing"`
	PrimaryAttemptNS int64  `json:"primary_attempt_ns"`
	WarmupPayloads   int    `json:"warmup_payloads"`
}
type client struct {
	http      *http.Client
	transport *http3.Transport
	quic      *quic.Transport
}

func newClient(daemonAddr string, r route, warmBytes, warmCount int) (*client, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	dc, topo, e := dialDaemon(ctx, daemonAddr)
	if e != nil {
		return nil, e
	}
	defer dc.Close()
	remote, e := snet.ParseUDPAddr(r.Remote)
	if e != nil {
		return nil, e
	}
	p, e := path.Choose(ctx, dc, remote.IA, path.WithSequence(r.Sequence))
	if e != nil {
		return nil, e
	}
	if p.Metadata().Fingerprint().String() != r.Fingerprint {
		return nil, fmt.Errorf("path fingerprint mismatch")
	}
	remote.Path = p.Dataplane()
	remote.NextHop = p.UnderlayNextHop()
	// An intra-AS path has no border-router next hop. This occurs when
	// separate logical proposers share the same lab AS.
	nextIP := remote.Host.IP
	if remote.NextHop != nil {
		nextIP = remote.NextHop.IP
	}
	local, e := addrutil.ResolveLocal(nextIP)
	if e != nil {
		return nil, e
	}
	network := &snet.SCIONNetwork{Topology: topo, SCMPHandler: ignoreSCMP{}}
	sock, e := network.Listen(ctx, "udp", &net.UDPAddr{IP: local})
	if e != nil {
		return nil, e
	}
	tr := &quic.Transport{Conn: sock}
	qc := quicConfig()
	qc.MaxIdleTimeout = connectionIdleTimeout
	rt := &http3.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true, NextProtos: []string{alpn}, ServerName: "e2e-http"}, QUICConfig: qc,
		Dial: func(ctx context.Context, _ string, tlsCfg *tls.Config, q *quic.Config) (*quic.Conn, error) {
			return tr.Dial(ctx, remote, tlsCfg, q)
		}}
	c := &client{http: &http.Client{Transport: rt}, transport: rt, quic: tr}
	req, _ := http.NewRequestWithContext(ctx, "GET", "https://e2e-http/ready", nil)
	resp, e := c.http.Do(req)
	if e != nil {
		c.close()
		return nil, e
	}
	_, e = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if e != nil || resp.StatusCode != 200 {
		c.close()
		return nil, fmt.Errorf("warmup failed: %v status=%d", e, resp.StatusCode)
	}
	for i := 0; i < warmCount; i++ {
		req, _ := http.NewRequestWithContext(ctx, "POST", "https://e2e-http/warmup", bytes.NewReader(bytes.Repeat([]byte{0xa5}, warmBytes)))
		resp, err := c.http.Do(req)
		if err != nil {
			c.close()
			return nil, err
		}
		_, err = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if err != nil || resp.StatusCode != 200 {
			c.close()
			return nil, fmt.Errorf("payload warmup failed")
		}
	}
	return c, nil
}
func (c *client) close()           { _ = c.transport.Close(); _ = c.quic.Close() }
func key(v string, r route) string { return v + "/" + r.Remote + "/" + r.Fingerprint }

func replay(configPath string) error {
	raw, e := os.ReadFile(configPath)
	if e != nil {
		return e
	}
	var cfg config
	if e = json.Unmarshal(raw, &cfg); e != nil {
		return e
	}
	if cfg.Payload <= 0 || cfg.Payload > 4*1024*1024 || cfg.IntervalNS <= cfg.DeadlineNS {
		return fmt.Errorf("invalid payload or overlapping rounds")
	}
	log := newLog(cfg.Output)
	defer log.f.Close()
	clients := map[string]*client{}
	defer func() {
		for _, c := range clients {
			c.close()
		}
	}()
	for _, j := range cfg.Jobs {
		for _, rs := range j.Targets {
			for _, r := range rs {
				k := key(j.Validator, r)
				if clients[k] != nil {
					continue
				}
				c, e := newClient(cfg.Daemon, r, cfg.Payload, cfg.WarmupPayloads)
				if e != nil {
					return fmt.Errorf("warmup %s: %w", k, e)
				}
				clients[k] = c
			}
		}
	}
	if e = save(cfg.Ready, map[string]any{"ready_ns": mono(), "clients": len(clients)}); e != nil {
		return e
	}
	var start struct {
		StartNS int64 `json:"start_ns"`
	}
	until := time.Now().Add(90 * time.Second)
	for {
		b, e := os.ReadFile(cfg.StartFile)
		if e == nil && json.Unmarshal(b, &start) == nil && start.StartNS > 0 {
			break
		}
		if time.Now().After(until) {
			return fmt.Errorf("start barrier timed out")
		}
		time.Sleep(20 * time.Millisecond)
	}
	if start.StartNS <= mono() {
		return fmt.Errorf("start barrier arrived late")
	}
	payload := bytes.Repeat([]byte{0xa5}, cfg.Payload)
	digest := sha256.Sum256(payload)
	hash := hex.EncodeToString(digest[:])
	for round := 0; round < cfg.Rounds; round++ {
		planned := start.StartNS + int64(round)*cfg.IntervalNS
		waitUntil(planned)
		var wg sync.WaitGroup
		for _, j := range cfg.Jobs {
			if j.Round != round {
				continue
			}
			j := j
			wg.Add(1)
			go func() {
				defer wg.Done()
				log.put(map[string]any{"kind": "start", "run": cfg.Run, "round": j.Round, "validator": j.Validator, "planned_ns": planned, "actual_ns": mono()})
				var targets sync.WaitGroup
				for _, rs := range j.Targets {
					rs := rs
					targets.Add(1)
					go func() {
						defer targets.Done()
						deadline := planned + cfg.DeadlineNS
						for attempt, r := range rs {
							if cfg.Routing == "surviving" && r.Core == cfg.FailedCore && len(rs) > 1 {
								continue
							}
							now := mono()
							if now >= deadline {
								break
							}
							remaining := deadline - now
							if cfg.Routing == "retry" && attempt == 0 && len(rs) > 1 && cfg.PrimaryAttemptNS < remaining {
								remaining = cfg.PrimaryAttemptNS
							}
							ctx, cancel := context.WithTimeout(context.Background(), time.Duration(remaining))
							req, _ := http.NewRequestWithContext(ctx, "POST", "https://e2e-http/proposal", bytes.NewReader(payload))
							for k, v := range map[string]string{"X-Run": cfg.Run, "X-Round": strconv.Itoa(j.Round), "X-Validator": j.Validator,
								"X-Attempt": strconv.Itoa(attempt), "X-Bytes": strconv.Itoa(cfg.Payload), "X-Hash": hash, "X-Path": r.Fingerprint} {
								req.Header.Set(k, v)
							}
							resp, err := clients[key(j.Validator, r)].http.Do(req)
							ok := false
							if err == nil {
								_, err = io.Copy(io.Discard, resp.Body)
								resp.Body.Close()
								ok = err == nil && resp.StatusCode == 200
							}
							event := map[string]any{"kind": "attempt", "run": cfg.Run, "round": j.Round, "validator": j.Validator,
								"receiver": r.Remote, "attempt": attempt, "path": r.Fingerprint, "start_ns": now, "end_ns": mono(), "acknowledged": ok,
								"application_bytes_offered": cfg.Payload}
							if err != nil {
								event["error"] = err.Error()
							}
							log.put(event)
							cancel()
							if ok {
								break
							}
							if cfg.Routing != "retry" {
								break
							}
						}
					}()
				}
				targets.Wait()
			}()
		}
		wg.Wait()
	}
	return save(filepath.Join(filepath.Dir(cfg.Output), "done.json"), map[string]any{"completed_ns": mono()})
}

func main() {
	mode := flag.String("mode", "clock", "clock, settings, serve, or replay")
	daemonAddr := flag.String("daemon", "", "SCION daemon")
	listen := flag.String("listen", "", "listen address")
	log := flag.String("log", "", "receiver log")
	cfg := flag.String("config", "", "replay config")
	flag.Parse()
	var e error
	switch *mode {
	case "clock":
		fmt.Println(mono())
	case "settings":
		e = json.NewEncoder(os.Stdout).Encode(map[string]any{
			"client_idle_timeout_seconds": connectionIdleTimeout.Seconds(),
			"server_idle_timeout_seconds": connectionIdleTimeout.Seconds(),
		})
	case "serve":
		e = serve(*daemonAddr, *listen, *log)
	case "replay":
		e = replay(*cfg)
	default:
		e = fmt.Errorf("unknown mode")
	}
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
