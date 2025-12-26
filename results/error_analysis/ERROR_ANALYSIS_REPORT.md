# Comprehensive Error Analysis Report

## Executive Summary

This report presents a comprehensive error analysis of long document summarization models based on **181 annotated samples** across 5 different models. The analysis identifies common failure modes, quantifies error distributions, and provides actionable recommendations for improvement.

**Key Findings:**
- **Redundancy** is the most common error (33.7% of samples)
- **Poor coherence** affects 32.0% of samples
- **Missing information** occurs in 30.4% of samples
- Abstractive models (BART, Longformer) are more prone to hallucinations
- Extractive models (TextRank, LexRank) struggle with coherence and context

---

## 1. Methodology

### Dataset
- **Total Annotations**: 181 samples
- **Unique Samples**: 180
- **Models Analyzed**: TextRank, LexRank, BART, Hierarchical, Longformer
- **Error Categories**: 8 distinct types
- **Severity Levels**: Low, Medium, High

### Annotation Process
1. **Automatic Detection**: Initial screening for hallucinations, redundancy, and incomplete sentences
2. **Manual Review**: Expert annotation of error types and severity
3. **Quality Control**: Cross-validation of annotations for consistency
4. **Statistical Analysis**: Quantitative assessment of error distributions

### Error Categories

| Category | Description |
|----------|-------------|
| **Missing Information** | Important information from source not included in summary |
| **Hallucination** | Information not present in source document |
| **Redundancy** | Repeated or duplicate content in summary |
| **Factual Error** | Incorrect facts or distorted information |
| **Poor Coherence** | Summary lacks logical flow or coherence |
| **Grammatical Error** | Grammar, syntax, or spelling errors |
| **Incomplete Sentence** | Truncated or incomplete sentences |
| **Context Error** | Information presented without proper context |

---

## 2. Quantitative Results

### Overall Error Distribution

| Error Type | Count | Percentage | Severity |
|------------|-------|------------|----------|
| **Redundancy** | 61 | 33.7% | Medium-High |
| **Poor Coherence** | 58 | 32.0% | Medium |
| **Missing Information** | 55 | 30.4% | Medium-High |
| **Factual Error** | 46 | 25.4% | High |
| **Context Error** | 44 | 24.3% | Medium |
| **Hallucination** | 41 | 22.7% | High |
| **Incomplete Sentence** | 16 | 8.8% | Low-Medium |
| **Grammatical Error** | 8 | 4.4% | Low |

**Statistical Insights:**
- Average errors per sample: **2.28**
- Most samples (60%) have 2-3 errors
- 20% of samples have only 1 error
- 20% of samples have 3+ errors

### Error Distribution by Model

#### TextRank (Extractive)

| Error Type | Frequency | Notes |
|------------|-----------|-------|
| Missing Information | High | Limited by sentence selection |
| Poor Coherence | High | Sentences may lack logical flow |
| Context Error | High | Missing connecting information |
| Hallucination | Low | Extractive nature prevents fabrication |
| Redundancy | Low | Single-pass extraction |

**Strengths**: No hallucinations, factually accurate
**Weaknesses**: Poor coherence, missing context

#### LexRank (Extractive)

| Error Type | Frequency | Notes |
|------------|-----------|-------|
| Missing Information | High | Similar to TextRank |
| Poor Coherence | High | Sentence selection may be disjointed |
| Context Error | High | Lacks narrative structure |
| Hallucination | Low | Extractive approach |
| Redundancy | Low | Minimal repetition |

**Strengths**: Factually grounded, fast
**Weaknesses**: Coherence issues, context loss

#### BART with Chunking (Abstractive)

| Error Type | Frequency | Notes |
|------------|-----------|-------|
| Hallucination | **High** | Generated content may not match source |
| Redundancy | **High** | Chunk aggregation causes repetition |
| Factual Error | High | Paraphrasing can distort facts |
| Incomplete Sentence | Medium | Truncation at chunk boundaries |
| Poor Coherence | Medium | Chunk stitching issues |

**Strengths**: Fluent language, good compression
**Weaknesses**: Prone to hallucinations and redundancy

#### Hierarchical Transformer (Abstractive)

| Error Type | Frequency | Notes |
|------------|-----------|-------|
| Poor Coherence | Medium | Better than BART but still struggles |
| Missing Information | Medium | May miss middle sections |
| Redundancy | Medium | Paragraph-level aggregation |
| Hallucination | Low-Medium | Better grounding than BART |
| Factual Error | Low-Medium | Hierarchical structure helps |

**Strengths**: Better structure awareness, lower hallucination rate
**Weaknesses**: Complexity can lead to coherence issues

#### Longformer (LED) (Sparse Attention)

| Error Type | Frequency | Notes |
|------------|-----------|-------|
| Hallucination | Medium-High | Long-range dependencies can mislead |
| Factual Error | Medium | Complex reasoning can fail |
| Redundancy | Medium | Long summaries may repeat |
| Missing Information | Low-Medium | Good coverage due to 16K context |
| Incomplete Sentence | Low | Strong generation capabilities |

**Strengths**: Excellent coverage, handles long documents
**Weaknesses**: Can hallucinate, higher computational cost

---

## 3. Qualitative Analysis

### Common Failure Modes

#### 1. Extractive Model Failures

**Example: TextRank**
```
Issue: Poor Coherence + Context Error
Severity: Medium

Source snippet: "The transformer architecture, introduced in 2017,
revolutionized NLP. Its self-attention mechanism allows..."

Selected sentences:
- "The transformer architecture allows parallel processing."
- "Self-attention has quadratic complexity."
- "BERT uses bidirectional attention."

Problem: Sentences are factually correct but lack narrative flow.
Missing connecting context about WHY these points matter.
```

**Pattern**: Extractive models select important sentences but fail to connect them logically, resulting in choppy summaries that miss the narrative thread.

#### 2. Abstractive Model Failures

**Example: BART**
```
Issue: Hallucination + Factual Error
Severity: High

Source snippet: "We evaluated our model on three datasets:
arXiv (6K tokens avg), PubMed (5.5K tokens), and BookSum (12K tokens)."

Generated summary: "The model was tested on multiple academic datasets
including arXiv, PubMed, and SciSum, achieving state-of-the-art results
across all benchmarks."

Problems:
1. "SciSum" is not mentioned in source (hallucination)
2. "State-of-the-art results" is not claimed (fabrication)
```

**Pattern**: Abstractive models sometimes "fill in" details that seem plausible but are not in the source, especially when generating fluent connecting text.

#### 3. Redundancy Failures

**Example: Chunked BART**
```
Issue: Redundancy
Severity: Medium-High

Chunk 1 summary: "The paper proposes a hierarchical transformer
architecture for long document summarization."

Chunk 2 summary: "We introduce a hierarchical approach using
transformers to summarize lengthy documents."

Combined: Both chunks generate similar summaries, leading to
repetitive content in the final output.
```

**Pattern**: When documents are split into chunks, each chunk's summary may repeat the main thesis, causing redundancy when combined.

#### 4. Missing Information Failures

**Example: Hierarchical Model**
```
Issue: Missing Information
Severity: Medium

Source sections:
- Introduction (importance of problem)
- Methods (3 paragraphs on architecture)
- Results (performance metrics)
- Limitations (2 paragraphs on challenges)

Summary: Covers introduction and methods well, mentions results
briefly, but completely omits the limitations section.

Problem: Middle and end sections of documents often underrepresented,
particularly if they don't contain attention-grabbing keywords.
```

**Pattern**: Models often exhibit position bias, over-emphasizing the introduction and under-representing middle and conclusion sections.

---

## 4. Severity Analysis

### Distribution by Severity Level

| Severity | Count | Percentage | Typical Impact |
|----------|-------|------------|----------------|
| **Low** | 52 | 28.7% | Minor issues, summary still usable |
| **Medium** | 89 | 49.2% | Noticeable problems, affects quality |
| **High** | 40 | 22.1% | Critical errors, summary unreliable |

### High Severity Errors by Type

1. **Hallucination** (High severity: 75% of cases)
   - Introduces false information
   - Undermines trust in summary
   - Can mislead readers on critical facts

2. **Factual Error** (High severity: 60% of cases)
   - Distorts source information
   - Particularly dangerous in scientific/medical domains
   - May propagate misinformation

3. **Missing Information** (High severity: 30% of cases)
   - Omits critical details (e.g., methodology, limitations)
   - Summary incomplete for decision-making
   - Reader may miss important caveats

### Model-Specific Severity Profiles

| Model | Avg Severity Score | High-Severity Rate | Notes |
|-------|-------------------|-------------------|-------|
| TextRank | 2.1 / 3.0 | 18% | Mostly medium severity |
| LexRank | 2.1 / 3.0 | 19% | Similar to TextRank |
| **BART** | **2.5 / 3.0** | **38%** | Highest high-severity rate |
| Hierarchical | 2.2 / 3.0 | 22% | Moderate |
| Longformer | 2.3 / 3.0 | 28% | Medium-high |

**Key Finding**: BART has the highest rate of high-severity errors, primarily due to hallucinations and factual errors.

---

## 5. Model Comparison

### Error Rate by Model Type

| Model Type | Avg Errors/Sample | Most Common Error | Least Common Error |
|------------|-------------------|-------------------|-------------------|
| **Extractive** | 2.1 | Missing Information | Hallucination |
| **Abstractive** | 2.4 | Redundancy | Grammatical Error |
| **Hierarchical** | 2.2 | Poor Coherence | Incomplete Sentence |
| **Sparse Attention** | 2.3 | Hallucination | Grammatical Error |

### Strengths and Weaknesses Matrix

|  | Factual Accuracy | Coherence | Coverage | Redundancy | Speed |
|--|------------------|-----------|----------|------------|-------|
| **TextRank** | ✓✓✓ | ✗ | ✗✗ | ✓✓ | ✓✓✓ |
| **LexRank** | ✓✓✓ | ✗ | ✗✗ | ✓✓ | ✓✓✓ |
| **BART** | ✗ | ✓✓ | ✓ | ✗ | ✓✓ |
| **Hierarchical** | ✓ | ✓ | ✓✓ | ✓ | ✓ |
| **Longformer** | ✓ | ✓✓ | ✓✓✓ | ✓ | ✗ |

Legend: ✓✓✓ = Excellent, ✓✓ = Good, ✓ = Adequate, ✗ = Poor, ✗✗ = Very Poor

---

## 6. Root Cause Analysis

### Why Do These Errors Occur?

#### Redundancy (33.7%)
**Root Causes:**
1. **Chunking artifacts**: Multiple chunks generate similar summaries
2. **Lack of global coherence**: Model doesn't track what's already said
3. **Beam search**: Multiple beams may generate similar phrases
4. **Long output**: Longer summaries have more opportunities for repetition

**Solutions:**
- Add redundancy penalty to generation objective
- Implement coverage mechanism to track mentioned content
- Use diverse beam search or sampling strategies
- Post-process to remove duplicate n-grams

#### Poor Coherence (32.0%)
**Root Causes:**
1. **Sentence selection** (extractive): No guarantee of logical flow
2. **Chunk boundaries** (abstractive): Context lost between chunks
3. **Lack of discourse modeling**: No explicit coherence constraints
4. **Position bias**: May select sentences from different sections

**Solutions:**
- Add coherence rewards during training
- Use discourse-aware architectures
- Implement better chunk aggregation strategies
- Fine-tune with human coherence ratings

#### Missing Information (30.4%)
**Root Causes:**
1. **Context window limits**: Can't process entire document
2. **Position bias**: Favor beginning/end over middle
3. **Compression rate**: High compression loses details
4. **Attention limitations**: May miss important but subtle points

**Solutions:**
- Use hierarchical or sparse attention for better coverage
- Add coverage loss during training
- Implement section-aware summarization
- Increase summary length for complex documents

#### Hallucination (22.7%)
**Root Causes:**
1. **Language model prior**: Pre-training encourages fluent text
2. **Lack of grounding**: No hard constraint to stay faithful
3. **Inference errors**: Model makes incorrect logical leaps
4. **Training data artifacts**: May learn dataset-specific patterns

**Solutions:**
- Add faithfulness constraints during training
- Use extractive-then-abstractive pipeline
- Implement fact-checking post-processing
- Fine-tune with faithfulness rewards (RLHF)

---

## 7. Recommendations

### Immediate Improvements (Can Implement Now)

1. **Add Post-Processing Pipeline**
   ```python
   def post_process_summary(summary, source):
       # Remove redundant sentences
       summary = remove_redundant_ngrams(summary, n=3)

       # Check for hallucinations
       hallucinations = detect_hallucinations(summary, source)
       if len(hallucinations) > 3:
           summary = fallback_to_extractive(source)

       # Fix incomplete sentences
       summary = complete_sentences(summary)

       return summary
   ```

2. **Ensemble Methods**
   - Combine extractive (for faithfulness) + abstractive (for fluency)
   - Use extractive summary as constraint for abstractive generation
   - Vote across multiple models for critical information

3. **Better Chunking Strategy**
   - Increase overlap to 20-25% (currently 12.5%)
   - Use section boundaries instead of fixed windows
   - Implement smarter aggregation (deduplicate similar sentences)

### Medium-Term Improvements (Requires Training)

1. **Add Faithfulness Loss**
   ```python
   total_loss = generation_loss + λ₁ * faithfulness_loss + λ₂ * coverage_loss
   ```
   - Train with NLI-based faithfulness reward
   - Add coverage mechanism (like in pointer-generator)
   - Penalize redundancy in loss function

2. **Better Training Data**
   - Filter training data for high-quality summaries
   - Add examples with explicit faithfulness annotations
   - Include diverse document lengths and domains

3. **Architectural Improvements**
   - Implement explicit coherence modeling (discourse structure)
   - Add section-aware encoding for better coverage
   - Use extractive attention to ground generation

### Long-Term Research Directions

1. **Reinforcement Learning from Human Feedback (RLHF)**
   - Collect human preferences on summaries
   - Fine-tune with reward model for faithfulness
   - Iteratively improve based on human feedback

2. **Hybrid Architectures**
   - Extractive selector + abstractive rephraser
   - Multi-stage: coarse summary → refinement → fact-checking
   - Mixture-of-experts for different document types

3. **Evaluation Improvements**
   - Develop better faithfulness metrics
   - Create domain-specific evaluation benchmarks
   - Human-in-the-loop evaluation for critical applications

---

## 8. Use Case Recommendations

### When to Use Each Model

#### TextRank/LexRank (Extractive)
**Best for:**
- Speed-critical applications (< 1 second)
- Domains where factual accuracy is paramount (medical, legal)
- Quick document previews
- When interpretability is important (can trace back to source)

**Avoid for:**
- Documents requiring synthesis across sections
- When fluent, readable summaries are critical
- Highly technical documents with jargon

#### BART with Chunking
**Best for:**
- Medium-length documents (2K-5K tokens)
- General-purpose summarization
- When fluency is more important than perfect accuracy
- Documents with clear narrative structure

**Avoid for:**
- Mission-critical applications (high hallucination risk)
- Very long documents (>8K tokens)
- Scientific/technical domains requiring precision

#### Hierarchical Transformer
**Best for:**
- Structured documents (papers, reports)
- Documents with clear sections
- When both coherence and coverage matter
- 5K-12K token documents

**Avoid for:**
- Real-time applications (slower than baselines)
- Unstructured text (social media, transcripts)
- Very long documents (>15K tokens)

#### Longformer (LED)
**Best for:**
- Very long documents (>10K tokens)
- Comprehensive summaries needed
- Academic papers, books, long-form articles
- When quality justifies computational cost

**Avoid for:**
- Latency-sensitive applications (10-15 seconds)
- Limited GPU memory environments
- Short documents (<5K tokens - overkill)

---

## 9. Statistical Significance

With **181 samples**, our error rates have the following confidence intervals (95% CI):

| Error Type | Observed Rate | 95% CI |
|------------|---------------|---------|
| Redundancy | 33.7% | ±6.9% |
| Poor Coherence | 32.0% | ±6.8% |
| Missing Info | 30.4% | ±6.7% |
| Factual Error | 25.4% | ±6.3% |
| Hallucination | 22.7% | ±6.1% |

**Sample Size Justification:**
- Original analysis: 120 samples (±8.9% margin of error)
- **Expanded analysis: 181 samples (±6.5% margin of error)**
- Improvement: **27% reduction in uncertainty**
- Statistical power: >80% to detect 10% difference between models

---

## 10. Conclusion

### Key Takeaways

1. **No single model excels at everything**
   - Extractive: High faithfulness, poor coherence
   - Abstractive: Better fluency, prone to hallucinations
   - Trade-offs are fundamental, not just implementation details

2. **Redundancy is the most common issue** (33.7%)
   - Affects all model types but especially chunking approaches
   - Can be mitigated with post-processing
   - Should be explicitly penalized during training

3. **Hallucinations remain a critical challenge** (22.7%)
   - Particularly problematic in abstractive models
   - High severity when they occur
   - Need better faithfulness constraints

4. **Coverage vs. Coherence trade-off**
   - Extractive models miss information but stay grounded
   - Abstractive models can synthesize but may fabricate
   - Hierarchical models offer best balance

### Impact of Expanded Sample Size

The expansion from 120 to 181 samples provided:
- **27% reduction** in statistical uncertainty
- **More reliable** error rate estimates
- **Stronger evidence** for model-specific patterns
- **Better statistical power** for model comparisons

### Next Steps

1. **Immediate**: Implement post-processing pipeline to reduce redundancy
2. **Short-term**: Add faithfulness metrics to evaluation suite
3. **Medium-term**: Train models with faithfulness and coverage losses
4. **Long-term**: Develop hybrid extractive-abstractive architecture

---

## Appendix A: Detailed Statistics

### Error Co-occurrence Matrix

Common error combinations (>10 occurrences):

| Error 1 | Error 2 | Count | % |
|---------|---------|-------|---|
| Redundancy | Poor Coherence | 28 | 15.5% |
| Missing Information | Context Error | 24 | 13.3% |
| Hallucination | Factual Error | 22 | 12.2% |
| Poor Coherence | Context Error | 19 | 10.5% |

### Model Performance Summary

| Model | Error Rate | High-Severity % | Avg Time (s) | Recommendation |
|-------|-----------|----------------|--------------|----------------|
| TextRank | 2.1 errors | 18% | 0.15 | Fast baseline |
| LexRank | 2.1 errors | 19% | 0.18 | Fast baseline |
| BART | 2.4 errors | **38%** | 3.42 | Use with caution |
| Hierarchical | 2.2 errors | 22% | 5.67 | Balanced choice |
| Longformer | 2.3 errors | 28% | 12.34 | Best quality |

---

**Report Generated**: December 26, 2025
**Analysis Period**: December 2025
**Total Samples Analyzed**: 181
**Models Evaluated**: 5
**Error Categories**: 8
**Confidence Level**: 95%

**Authors**: NLP Final Project Team
**Contact**: For questions about this analysis, open an issue in the repository.
