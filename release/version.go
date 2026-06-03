// SPDX-License-Identifier: MIT OR Apache-2.0

// Package release holds the GitHub release-picker logic used by the admin
// panel's update check: a faithful PEP 440 version parser/comparator and the
// stable-release selector built on it. It is a leaf package with no network or
// MLX coupling, so the version semantics stay unit-testable on their own.
package release

import (
	"regexp"
	"strconv"
	"strings"
)

// versionPattern is the canonical PEP 440 grammar (the same one the reference's
// packaging library uses), inlined without the VERBOSE whitespace and with a
// leading (?i) for the case-insensitive match. A leading "v" and surrounding
// whitespace are tolerated.
var versionPattern = regexp.MustCompile(`(?i)^\s*v?(?:(?:(?P<epoch>[0-9]+)!)?(?P<release>[0-9]+(?:\.[0-9]+)*)(?P<pre>[-_\.]?(?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)[-_\.]?(?P<pre_n>[0-9]+)?)?(?P<post>(?:-(?P<post_n1>[0-9]+))|(?:[-_\.]?(?P<post_l>post|rev|r)[-_\.]?(?P<post_n2>[0-9]+)?))?(?P<dev>[-_\.]?(?P<dev_l>dev)[-_\.]?(?P<dev_n>[0-9]+)?)?)(?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?\s*$`)

// letterNum is a normalized (letter, number) release qualifier, e.g. the pre
// segment ("rc", 1) or post segment ("post", 2).
type letterNum struct {
	letter string
	num    int
}

// localElem is one dot/dash/underscore-separated component of a local version
// label. Numeric components compare as integers and sort after string
// components, matching PEP 440.
type localElem struct {
	isInt bool
	num   int
	str   string
}

// Version is a parsed PEP 440 version.
type Version struct {
	epoch   int
	release []int
	pre     *letterNum
	post    *letterNum
	dev     *letterNum
	local   []localElem // nil means no local segment
	hasLoc  bool
}

// ParseVersion parses a PEP 440 version string, reporting ok=false for an
// unparseable value (the reference's InvalidVersion).
func ParseVersion(s string) (Version, bool) {
	m := versionPattern.FindStringSubmatch(s)
	if m == nil {
		return Version{}, false
	}
	g := func(name string) string {
		i := versionPattern.SubexpIndex(name)
		if i < 0 {
			return ""
		}
		return m[i]
	}

	v := Version{}
	if e := g("epoch"); e != "" {
		v.epoch = atoi(e)
	}
	for part := range strings.SplitSeq(g("release"), ".") {
		v.release = append(v.release, atoi(part))
	}
	v.pre = parseLetterVersion(g("pre_l"), g("pre_n"))
	postNum := g("post_n1")
	if postNum == "" {
		postNum = g("post_n2")
	}
	v.post = parseLetterVersion(g("post_l"), postNum)
	v.dev = parseLetterVersion(g("dev_l"), g("dev_n"))
	if loc := g("local"); loc != "" {
		v.hasLoc = true
		v.local = parseLocal(loc)
	}
	return v, true
}

// parseLetterVersion normalizes a (letter, number) qualifier the way packaging
// does: alpha/beta spellings collapse to a/b, c/pre/preview to rc, rev/r to
// post, a missing number defaults to 0, and a bare number with no letter (the
// "-N" implicit-post form) becomes a post segment.
func parseLetterVersion(letter, number string) *letterNum {
	if letter != "" {
		num := 0
		if number != "" {
			num = atoi(number)
		}
		l := strings.ToLower(letter)
		switch l {
		case "alpha":
			l = "a"
		case "beta":
			l = "b"
		case "c", "pre", "preview":
			l = "rc"
		case "rev", "r":
			l = "post"
		}
		return &letterNum{letter: l, num: num}
	}
	if number != "" {
		return &letterNum{letter: "post", num: atoi(number)}
	}
	return nil
}

// parseLocal splits a local label on ./-/_ and lowercases non-numeric parts.
func parseLocal(local string) []localElem {
	var out []localElem
	for part := range strings.FieldsFuncSeq(local, func(r rune) bool {
		return r == '.' || r == '-' || r == '_'
	}) {
		if isAllDigits(part) {
			out = append(out, localElem{isInt: true, num: atoi(part)})
		} else {
			out = append(out, localElem{str: strings.ToLower(part)})
		}
	}
	return out
}

// IsPrerelease reports whether the version is a pre-release: it has a pre or dev
// segment. A post-only release is not a pre-release.
func (v Version) IsPrerelease() bool {
	return v.pre != nil || v.dev != nil
}

// String renders the normalized PEP 440 form, matching packaging's str().
func (v Version) String() string {
	var b strings.Builder
	if v.epoch != 0 {
		b.WriteString(strconv.Itoa(v.epoch))
		b.WriteByte('!')
	}
	for i, r := range v.release {
		if i > 0 {
			b.WriteByte('.')
		}
		b.WriteString(strconv.Itoa(r))
	}
	if v.pre != nil {
		b.WriteString(v.pre.letter)
		b.WriteString(strconv.Itoa(v.pre.num))
	}
	if v.post != nil {
		b.WriteString(".post")
		b.WriteString(strconv.Itoa(v.post.num))
	}
	if v.dev != nil {
		b.WriteString(".dev")
		b.WriteString(strconv.Itoa(v.dev.num))
	}
	if v.hasLoc {
		b.WriteByte('+')
		for i, e := range v.local {
			if i > 0 {
				b.WriteByte('.')
			}
			if e.isInt {
				b.WriteString(strconv.Itoa(e.num))
			} else {
				b.WriteString(e.str)
			}
		}
	}
	return b.String()
}

// Compare returns -1, 0, or 1 as v sorts before, equal to, or after other, using
// the PEP 440 ordering (epoch, release, pre, post, dev, local) with trailing
// zeros stripped from the release segment.
func (v Version) Compare(other Version) int {
	if c := cmpInt(v.epoch, other.epoch); c != 0 {
		return c
	}
	if c := cmpInts(stripTrailingZeros(v.release), stripTrailingZeros(other.release)); c != 0 {
		return c
	}
	if c := cmpPre(v, other); c != 0 {
		return c
	}
	if c := cmpPost(v.post, other.post); c != 0 {
		return c
	}
	if c := cmpDev(v.dev, other.dev); c != 0 {
		return c
	}
	return cmpLocal(v, other)
}

// preRank maps the pre segment to its sort sentinel: a dev-only release sorts
// before any pre-release (-1), a final release sorts after them (+1), and an
// actual pre-release (0) compares by its (letter, number).
func preRank(v Version) int {
	switch {
	case v.pre == nil && v.post == nil && v.dev != nil:
		return -1
	case v.pre == nil:
		return 1
	default:
		return 0
	}
}

func cmpPre(a, b Version) int {
	ra, rb := preRank(a), preRank(b)
	if ra != rb {
		return cmpInt(ra, rb)
	}
	if ra != 0 {
		return 0
	}
	if c := strings.Compare(a.pre.letter, b.pre.letter); c != 0 {
		return c
	}
	return cmpInt(a.pre.num, b.pre.num)
}

// cmpPost compares post segments: absent sorts before present, two present
// segments compare by number (the letter is always "post").
func cmpPost(a, b *letterNum) int {
	ra, rb := boolRank(a != nil, -1), boolRank(b != nil, -1)
	if ra != rb {
		return cmpInt(ra, rb)
	}
	if a == nil {
		return 0
	}
	return cmpInt(a.num, b.num)
}

// cmpDev compares dev segments: a present dev sorts before an absent one.
func cmpDev(a, b *letterNum) int {
	ra, rb := boolRank(a != nil, 1), boolRank(b != nil, 1)
	if ra != rb {
		return cmpInt(ra, rb)
	}
	if a == nil {
		return 0
	}
	return cmpInt(a.num, b.num)
}

// cmpLocal compares local segments: absent sorts before present, then element
// by element with numeric components sorting after string components.
func cmpLocal(a, b Version) int {
	if a.hasLoc != b.hasLoc {
		return boolRank(a.hasLoc, -1) - boolRank(b.hasLoc, -1)
	}
	if !a.hasLoc {
		return 0
	}
	for i := 0; i < len(a.local) && i < len(b.local); i++ {
		if c := cmpLocalElem(a.local[i], b.local[i]); c != 0 {
			return c
		}
	}
	return cmpInt(len(a.local), len(b.local))
}

func cmpLocalElem(a, b localElem) int {
	switch {
	case a.isInt && b.isInt:
		return cmpInt(a.num, b.num)
	case a.isInt && !b.isInt:
		return 1 // string component carries -inf in its first tuple slot
	case !a.isInt && b.isInt:
		return -1
	default:
		return strings.Compare(a.str, b.str)
	}
}

// boolRank returns present (0) or the given absent rank.
func boolRank(present bool, absent int) int {
	if present {
		return 0
	}
	return absent
}

func stripTrailingZeros(r []int) []int {
	n := len(r)
	for n > 0 && r[n-1] == 0 {
		n--
	}
	return r[:n]
}

func cmpInts(a, b []int) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if c := cmpInt(a[i], b[i]); c != 0 {
			return c
		}
	}
	return cmpInt(len(a), len(b))
}

func cmpInt(a, b int) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

func atoi(s string) int {
	n, _ := strconv.Atoi(s)
	return n
}

func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := range len(s) {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}
