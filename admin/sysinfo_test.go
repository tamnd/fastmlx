// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type sysinfoDiskIn struct {
	TotalBytes int  `json:"total_bytes"`
	Ok         bool `json:"ok"`
}

type sysinfoDiskCase struct {
	In  sysinfoDiskIn  `json:"in"`
	Out map[string]any `json:"out"`
}

type sysinfoMemIn struct {
	TotalBytes int `json:"total_bytes"`
	Available  int `json:"available"`
	Phys       int `json:"phys"`
	Iogpu      int `json:"iogpu"`
	WiredReq   int `json:"wired_req"`
	Free       int `json:"free"`
	Inactive   int `json:"inactive"`
	Active     int `json:"active"`
}

type sysinfoMemCase struct {
	In  sysinfoMemIn   `json:"in"`
	Out map[string]any `json:"out"`
}

type sysinfoFixture struct {
	Disk   []sysinfoDiskCase `json:"disk"`
	Memory []sysinfoMemCase  `json:"memory"`
}

func loadSysinfo(t *testing.T) sysinfoFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/sysinfo.json")
	if err != nil {
		t.Fatal(err)
	}
	var f sysinfoFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestSsdDiskInfoParity(t *testing.T) {
	for i, c := range loadSysinfo(t).Disk {
		got := jsonRoundTrip(t, SsdDiskInfo(c.In.TotalBytes, c.In.Ok))
		want := jsonRoundTrip(t, c.Out)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("SsdDiskInfo case %d:\n got  %v\n want %v", i, got, want)
		}
	}
}

func TestSystemMemoryInfoParity(t *testing.T) {
	for i, c := range loadSysinfo(t).Memory {
		in := SystemMemoryInputs{
			TotalBytes:             c.In.TotalBytes,
			AvailableBytes:         c.In.Available,
			PhysFootprintBytes:     c.In.Phys,
			IogpuWiredLimitBytes:   c.In.Iogpu,
			WiredLimitRequestBytes: c.In.WiredReq,
			FreeMemoryBytes:        c.In.Free,
			InactiveMemoryBytes:    c.In.Inactive,
			ActiveMemoryBytes:      c.In.Active,
		}
		got := jsonRoundTrip(t, SystemMemoryInfo(in))
		want := jsonRoundTrip(t, c.Out)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("SystemMemoryInfo case %d:\n got  %v\n want %v", i, got, want)
		}
	}
}

func BenchmarkSystemMemoryInfo(b *testing.B) {
	in := SystemMemoryInputs{
		TotalBytes:             17179869184,
		AvailableBytes:         8589934592,
		PhysFootprintBytes:     4294967296,
		IogpuWiredLimitBytes:   12884901888,
		WiredLimitRequestBytes: 10737418240,
		FreeMemoryBytes:        2147483648,
		InactiveMemoryBytes:    1073741824,
		ActiveMemoryBytes:      3221225472,
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = SystemMemoryInfo(in)
	}
}
