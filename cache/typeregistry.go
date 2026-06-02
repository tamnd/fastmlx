// SPDX-License-Identifier: MIT OR Apache-2.0

package cache

// CacheType names a KV-cache layout. The string values match the cache class
// names a restored cache serializes (type(cache).__name__), so the registry can
// route a serialized name back to the right handler.
type CacheType string

const (
	TypeKVCache              CacheType = "KVCache"
	TypeRotatingKVCache      CacheType = "RotatingKVCache"
	TypeBatchKVCache         CacheType = "BatchKVCache"
	TypeBatchRotatingKVCache CacheType = "BatchRotatingKVCache"
	TypeArraysCache          CacheType = "ArraysCache"
	TypeQuantizedKVCache     CacheType = "QuantizedKVCache"
	TypeCacheList            CacheType = "CacheList"
	TypePoolingCache         CacheType = "PoolingCache"
	TypeBatchPoolingCache    CacheType = "BatchPoolingCache"
)

// classNameEntry pairs a serialized class name with the cache type it routes to.
// The order is the upstream insertion order, which matters for the reverse
// lookup (the first class name mapped to a type wins, e.g. KVCache over the
// TurboQuant aliases that also map to KVCACHE).
type classNameEntry struct {
	name string
	ct   CacheType
}

// cacheClassNames maps serialized cache class names to their cache type. Several
// names collapse onto a base type: the rotating subclass that clamps size()
// routes through RotatingKVCache, and the TurboQuant variants route through
// KVCache so they read as block-sliceable (prefix-cache checks their class name
// first for the TurboQuant-specific path). The pooling and batch-rotating names
// have no registered handler and fall through to the default handler.
var cacheClassNames = []classNameEntry{
	{"KVCache", TypeKVCache},
	{"RotatingKVCache", TypeRotatingKVCache},
	{"PrefillReadyRotatingKVCache", TypeRotatingKVCache},
	{"BatchKVCache", TypeBatchKVCache},
	{"BatchRotatingKVCache", TypeBatchRotatingKVCache},
	{"ArraysCache", TypeArraysCache},
	{"QuantizedKVCache", TypeQuantizedKVCache},
	{"CacheList", TypeCacheList},
	{"TurboQuantKVCache", TypeKVCache},
	{"BatchTurboQuantKVCache", TypeKVCache},
	{"PoolingCache", TypePoolingCache},
	{"BatchPoolingCache", TypeBatchPoolingCache},
}

var cacheClassNameMap = func() map[string]CacheType {
	m := make(map[string]CacheType, len(cacheClassNames))
	for _, e := range cacheClassNames {
		m[e.name] = e.ct
	}
	return m
}()

// registeredSlicing is the block-slicing capability of the four handlers the
// registry installs on init. Every other cache type falls through to the
// default handler, which subclasses the KVCache handler and so slices.
var registeredSlicing = map[CacheType]bool{
	TypeKVCache:         true,
	TypeRotatingKVCache: false, // cannot safely slice a rotating cache
	TypeArraysCache:     false, // generic arrays may not be sequence-indexed
	TypeCacheList:       false, // mixed sub-cache types prevent slicing
}

// DefaultSlicing is the default handler's block-slicing capability (it derives
// from the KVCache handler).
const DefaultSlicing = true

// HandlerSupportsBlockSlicing reports whether the handler for a cache type can
// slice a cache by sequence position. Unregistered types use the default
// handler.
func HandlerSupportsBlockSlicing(ct CacheType) bool {
	if v, ok := registeredSlicing[ct]; ok {
		return v
	}
	return DefaultSlicing
}

// CacheTypeForClassName resolves a serialized cache class name to its cache
// type. SizedArraysCache is a wrapper that resolves to ArraysCache. The second
// result is false for an unknown name (which the caller routes to the default
// handler).
func CacheTypeForClassName(className string) (CacheType, bool) {
	if className == "SizedArraysCache" {
		return TypeArraysCache, true
	}
	ct, ok := cacheClassNameMap[className]
	return ct, ok
}

// IsSliceableByClassName reports whether a cache identified only by its class
// name supports block slicing. An unknown name uses the default handler.
func IsSliceableByClassName(className string) bool {
	ct, ok := CacheTypeForClassName(className)
	if !ok {
		return DefaultSlicing
	}
	return HandlerSupportsBlockSlicing(ct)
}

// ClassNameForType returns a representative class name for a cache type, the
// first name mapped to it in upstream order. A type with no mapped name returns
// its own string value.
func ClassNameForType(ct CacheType) string {
	for _, e := range cacheClassNames {
		if e.ct == ct {
			return e.name
		}
	}
	return string(ct)
}

// KnownClassNames returns the serialized class names the registry recognizes,
// in upstream order.
func KnownClassNames() []string {
	names := make([]string, len(cacheClassNames))
	for i, e := range cacheClassNames {
		names[i] = e.name
	}
	return names
}
