// SPDX-License-Identifier: MIT OR Apache-2.0

// Package server wires the net/http application: lifespan (bind, engine start,
// readiness), middleware (request-id, access logging, CORS, auth), exception
// handlers, and graceful shutdown. It sets up the listener, the route
// lifespan, minus the framework.
package server

import (
	"context"
	"errors"
	"log"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/tamnd/fastmlx/enginecore"
	"github.com/tamnd/fastmlx/routes"
)

// Config configures the HTTP application.
type Config struct {
	Addr        string                    // host:port
	Engine      *enginecore.BatchedEngine // the (mock-backed) LLM engine
	APIKeys     []string                  // empty disables auth
	CORSOrigins []string                  // "*" allows any origin
	Logger      *log.Logger
}

// App is the running server: an engine, the route table, and bind-time state.
type App struct {
	cfg    Config
	router *routes.Router
	logger *log.Logger

	mu    sync.RWMutex
	ready bool

	httpServer *http.Server
}

// NewApp builds an application over a started-or-startable engine.
func NewApp(cfg Config) *App {
	if cfg.Logger == nil {
		cfg.Logger = log.Default()
	}
	return &App{
		cfg:    cfg,
		router: routes.New(cfg.Engine),
		logger: cfg.Logger,
	}
}

// Handler builds the full middleware chain around the route mux. CORS sits
// outside auth so preflight requests are answered without a key.
func (a *App) Handler() http.Handler {
	mux := http.NewServeMux()
	a.router.Register(mux)

	var h http.Handler = mux
	h = a.withAuth(h)
	h = a.withCORS(h)
	h = a.withRecover(h)
	h = a.withRequestLog(h)
	return h
}

// Run binds the socket (before any slow model load so the port is held),
// starts the engine, serves, and shuts down gracefully when ctx is cancelled.
func (a *App) Run(ctx context.Context) error {
	ln, err := net.Listen("tcp", a.cfg.Addr)
	if err != nil {
		return err
	}

	a.cfg.Engine.Start(ctx)

	a.httpServer = &http.Server{
		Handler:           a.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	a.setReady(true)
	a.logger.Printf("fastmlx listening on %s (model %q)", ln.Addr(), a.cfg.Engine.ModelName())

	serveErr := make(chan error, 1)
	go func() { serveErr <- a.httpServer.Serve(ln) }()

	select {
	case <-ctx.Done():
		return a.shutdown()
	case err := <-serveErr:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	}
}

func (a *App) shutdown() error {
	a.setReady(false)
	shutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	err := a.httpServer.Shutdown(shutCtx)
	a.cfg.Engine.Stop()
	a.logger.Printf("fastmlx stopped")
	return err
}

func (a *App) setReady(v bool) {
	a.mu.Lock()
	a.ready = v
	a.mu.Unlock()
}

// Ready reports whether the server is accepting traffic.
func (a *App) Ready() bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.ready
}
