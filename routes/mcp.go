// SPDX-License-Identifier: MIT OR Apache-2.0

package routes

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/tamnd/fastmlx/mcp"
)

// mcpManager is the subset of the MCP manager the routes use. *mcp.Manager
// satisfies it; a nil manager means MCP was not configured.
type mcpManager interface {
	AllTools() []mcp.Tool
	ServerStatuses() []mcp.ServerStatus
	ExecuteTool(ctx context.Context, fullName string, arguments json.RawMessage, timeout float64) mcp.ToolResult
}

// SetMCPManager attaches an MCP manager so the /v1/mcp/* routes can serve tool
// discovery and execution. Without one those routes report MCP as unconfigured.
func (rt *Router) SetMCPManager(m mcpManager) { rt.mcp = m }

// mcpToolInfo describes one discovered tool.
type mcpToolInfo struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Server      string          `json:"server"`
	Parameters  json.RawMessage `json:"parameters"`
}

type mcpToolsResponse struct {
	Tools []mcpToolInfo `json:"tools"`
	Count int           `json:"count"`
}

type mcpServerInfo struct {
	Name       string  `json:"name"`
	State      string  `json:"state"`
	Transport  string  `json:"transport"`
	ToolsCount int     `json:"tools_count"`
	Error      *string `json:"error"`
}

type mcpServersResponse struct {
	Servers []mcpServerInfo `json:"servers"`
}

type mcpExecuteRequest struct {
	ToolName  string          `json:"tool_name"`
	Tool      string          `json:"tool"`
	Arguments json.RawMessage `json:"arguments"`
}

type mcpExecuteResponse struct {
	ToolName     string          `json:"tool_name"`
	Content      json.RawMessage `json:"content"`
	IsError      bool            `json:"is_error"`
	ErrorMessage *string         `json:"error_message"`
}

// MCPTools handles GET /v1/mcp/tools. With no manager it returns an empty list.
func (rt *Router) MCPTools(w http.ResponseWriter, r *http.Request) {
	if rt.mcp == nil {
		writeJSON(w, http.StatusOK, mcpToolsResponse{Tools: []mcpToolInfo{}, Count: 0})
		return
	}
	tools := rt.mcp.AllTools()
	infos := make([]mcpToolInfo, 0, len(tools))
	for _, t := range tools {
		params := t.InputSchema
		if len(params) == 0 {
			params = json.RawMessage("{}")
		}
		infos = append(infos, mcpToolInfo{
			Name:        t.FullName(),
			Description: t.Description,
			Server:      t.ServerName,
			Parameters:  params,
		})
	}
	writeJSON(w, http.StatusOK, mcpToolsResponse{Tools: infos, Count: len(infos)})
}

// MCPServers handles GET /v1/mcp/servers. With no manager it returns an empty
// list.
func (rt *Router) MCPServers(w http.ResponseWriter, r *http.Request) {
	if rt.mcp == nil {
		writeJSON(w, http.StatusOK, mcpServersResponse{Servers: []mcpServerInfo{}})
		return
	}
	statuses := rt.mcp.ServerStatuses()
	infos := make([]mcpServerInfo, 0, len(statuses))
	for _, s := range statuses {
		var errPtr *string
		if s.Error != "" {
			e := s.Error
			errPtr = &e
		}
		infos = append(infos, mcpServerInfo{
			Name:       s.Name,
			State:      string(s.State),
			Transport:  string(s.Transport),
			ToolsCount: s.ToolsCount,
			Error:      errPtr,
		})
	}
	writeJSON(w, http.StatusOK, mcpServersResponse{Servers: infos})
}

// MCPExecute handles POST /v1/mcp/execute. With no manager it reports MCP as
// unconfigured (503), matching the reference.
func (rt *Router) MCPExecute(w http.ResponseWriter, r *http.Request) {
	if rt.mcp == nil {
		writeError(w, http.StatusServiceUnavailable,
			"MCP not configured. Start server with --mcp-config", "service_unavailable_error", "")
		return
	}
	var req mcpExecuteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error(), "invalid_request_error", "")
		return
	}
	name := req.ToolName
	if name == "" {
		name = req.Tool // "tool" is accepted as an alias for "tool_name"
	}

	res := rt.mcp.ExecuteTool(r.Context(), name, req.Arguments, 0)

	var errPtr *string
	if res.ErrorMessage != "" {
		e := res.ErrorMessage
		errPtr = &e
	}
	content := res.Content
	if len(content) == 0 {
		content = json.RawMessage("null")
	}
	writeJSON(w, http.StatusOK, mcpExecuteResponse{
		ToolName:     res.ToolName,
		Content:      content,
		IsError:      res.IsError,
		ErrorMessage: errPtr,
	})
}
