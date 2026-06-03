// SPDX-License-Identifier: MIT OR Apache-2.0

// Package netutil holds the GPU-free network-alias logic: validating that a
// string is a usable DNS hostname or routable IP address, and assembling the
// ordered, de-duplicated list of aliases a running server answers to. The live
// socket and interface probes that discover the hostname, FQDN, and local IPs
// are the system seam, injected as plain inputs so the assembly is
// deterministic and fully testable.
package netutil

import (
	"net/netip"
	"strings"
)

// IsValidHostname reports whether value looks like a valid DNS hostname. A
// single trailing dot is allowed; each dot-separated label must be 1 to 63
// ASCII letters, digits, or hyphens and may not start or end with a hyphen. The
// whole name may not exceed 253 characters.
//
// The reference matches each label against ^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$; the
// lookarounds are reproduced directly since Go's regexp engine has none.
func IsValidHostname(value string) bool {
	if value == "" || len([]rune(value)) > 253 {
		return false
	}
	candidate := strings.TrimSuffix(value, ".")
	for label := range strings.SplitSeq(candidate, ".") {
		if !validLabel(label) {
			return false
		}
	}
	return true
}

// validLabel checks a single hostname label. Labels are ASCII-only, so a label
// containing any multi-byte rune fails the byte-range check and byte length
// equals rune length for the ones that pass.
func validLabel(label string) bool {
	n := len(label)
	if n < 1 || n > 63 {
		return false
	}
	if label[0] == '-' || label[n-1] == '-' {
		return false
	}
	for i := range n {
		c := label[i]
		switch {
		case c >= 'A' && c <= 'Z':
		case c >= 'a' && c <= 'z':
		case c >= '0' && c <= '9':
		case c == '-':
		default:
			return false
		}
	}
	return true
}

// IsValidIP reports whether value is a usable IPv4 or IPv6 alias address. The
// unspecified bind addresses 0.0.0.0 and :: parse as valid IPs but are rejected
// here since they are not routable as client-facing URL hosts.
func IsValidIP(value string) bool {
	ip, ok := parseIP(value)
	if !ok {
		return false
	}
	return !ip.IsUnspecified()
}

// parseIP parses a bare IPv4 or IPv6 address the way Python's
// ipaddress.ip_address does: leading zeros and zone identifiers are rejected.
func parseIP(value string) (netip.Addr, bool) {
	addr, err := netip.ParseAddr(value)
	if err != nil || addr.Zone() != "" {
		return netip.Addr{}, false
	}
	return addr, true
}

// IsValidAlias validates that value is a hostname or routable IP address. If the
// value parses as an IP at all, the IP check is authoritative: an IP-shaped
// string like 0.0.0.0 is rejected outright rather than slipping through as a
// hostname (digit-only labels being legal). Surrounding whitespace is trimmed
// first.
func IsValidAlias(value string) bool {
	value = strings.TrimSpace(value)
	if value == "" {
		return false
	}
	if _, ok := parseIP(value); ok {
		return IsValidIP(value)
	}
	return IsValidHostname(value)
}

// SystemAliases carries the system-probed inputs to DetectServerAliases: the
// values the reference reads from socket.gethostname, socket.getfqdn, and the
// local interface enumeration. Empty strings mark a probe that returned nothing.
// LocalIPv4 is expected to already exclude loopback addresses, matching the
// reference's _local_ipv4_addresses filter.
type SystemAliases struct {
	Hostname  string
	FQDN      string
	LocalIPv4 []string
}

// DetectServerAliases builds the ordered, de-duplicated list of aliases the
// server answers to. Order favors commonly accessible names: localhost (only
// when bound to loopback or all interfaces), the hostname, its mDNS .local form,
// the FQDN (reverse-DNS PTR records skipped), then any non-loopback IPv4
// addresses. The list is filtered to valid aliases and de-duplicated last.
func DetectServerAliases(host string, sys SystemAliases) []string {
	var candidates []string

	switch host {
	case "127.0.0.1", "localhost", "0.0.0.0", "::":
		candidates = append(candidates, "localhost", "127.0.0.1")
	}

	if sys.Hostname != "" {
		candidates = append(candidates, sys.Hostname)
		if !strings.HasSuffix(sys.Hostname, ".local") {
			candidates = append(candidates, sys.Hostname+".local")
		}
	}

	if sys.FQDN != "" && !strings.HasSuffix(sys.FQDN, ".ip6.arpa") && !strings.HasSuffix(sys.FQDN, ".in-addr.arpa") {
		candidates = append(candidates, sys.FQDN)
	}

	candidates = append(candidates, sys.LocalIPv4...)

	var valid []string
	for _, c := range candidates {
		if IsValidAlias(c) {
			valid = append(valid, c)
		}
	}
	return dedupePreserveOrder(valid)
}

// dedupePreserveOrder drops empty strings and repeats while keeping first-seen
// order.
func dedupePreserveOrder(items []string) []string {
	seen := make(map[string]bool, len(items))
	result := []string{}
	for _, item := range items {
		if item != "" && !seen[item] {
			seen[item] = true
			result = append(result, item)
		}
	}
	return result
}
