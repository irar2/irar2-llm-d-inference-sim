/*
Copyright 2026 The llm-d-inference-sim Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package llmdinferencesim

import (
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// ptrInt64 and ptrInt return pointers to their argument; small helpers for
// building request fixtures without leaking temp vars into each test.
func ptrInt64(v int64) *int64 { return &v }
func ptrInt(v int) *int       { return &v }

// newTextCompletionsFixture builds a TextCompletionsRequest populated with a
// cross-section of fields that duplicateWithPrompt must preserve when splitting
// an array-prompt request. Keep this exhaustive — if you add a field to the
// request type that should survive the copy, add it here too.
func newTextCompletionsFixture() *TextCompletionsRequest {
	req := &TextCompletionsRequest{}
	req.RequestID = "req-abc"
	req.Model = "test-model"
	req.DisplayedModel = "test-model-alias"
	req.Stream = true
	req.IgnoreEOS = true
	req.StreamOptions = &openaiserverapi.StreamOptions{IncludeUsage: true}
	threshold := 0.25
	req.CacheHitThreshold = &threshold
	req.Prompt = openaiserverapi.NewStringOrArrayFromSlice([]string{"one", "two"})
	req.MaxTokens = ptrInt64(42)
	req.Logprobs = ptrInt(3)
	return req
}

var _ = Describe("duplicateWithPrompt", func() {
	It("returns a fresh request with a single-string prompt and the new RequestID", func() {
		orig := newTextCompletionsFixture()

		dup := orig.duplicateWithPrompt("just-this", "req-abc-1").(*TextCompletionsRequest)

		Expect(dup).NotTo(BeIdenticalTo(orig))
		Expect(dup.GetRequestID()).To(Equal("req-abc-1"))
		Expect(dup.Prompt.IsArray()).To(BeFalse())
		Expect(dup.Prompt.String()).To(Equal("just-this"))
	})

	It("preserves non-prompt fields from the original", func() {
		orig := newTextCompletionsFixture()

		dup := orig.duplicateWithPrompt("x", "req-abc-0").(*TextCompletionsRequest)

		Expect(dup.GetModel()).To(Equal(orig.GetModel()))
		Expect(dup.GetDisplayedModel()).To(Equal(orig.GetDisplayedModel()))
		Expect(dup.IsStream()).To(Equal(orig.IsStream()))
		Expect(dup.IncludeUsage()).To(Equal(orig.IncludeUsage()))
		Expect(dup.GetIgnoreEOS()).To(Equal(orig.GetIgnoreEOS()))
		Expect(dup.ExtractMaxTokens()).To(Equal(orig.ExtractMaxTokens()))
		Expect(dup.GetLogprobs()).To(Equal(orig.GetLogprobs()))
		Expect(dup.GetCacheHitThreshold()).To(Equal(orig.GetCacheHitThreshold()))
	})

	It("does not mutate the original request", func() {
		orig := newTextCompletionsFixture()
		origPromptArr := orig.Prompt.Array()
		origID := orig.GetRequestID()

		_ = orig.duplicateWithPrompt("other", "req-abc-9")

		Expect(orig.GetRequestID()).To(Equal(origID))
		Expect(orig.Prompt.IsArray()).To(BeTrue())
		Expect(orig.Prompt.Array()).To(Equal(origPromptArr))
	})
})

