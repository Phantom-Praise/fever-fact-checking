# Proposed prompt revision — awaiting user confirmation

Status: proposal only. No classifier/prompt code has been changed and no new Llama calls have been made for this proposal.

BEFORE.txt is copied from an actual saved RAG request. PROPOSED_AFTER.txt contains general instructions, with no examples or vocabulary drawn from the six errors.

Proposed software aggregation, after validating the response format and citation IDs:

1. Any essential part REFUTES → final REFUTES.
2. Otherwise, every part SUPPORTS → final SUPPORTS.
3. Otherwise → final NOT ENOUGH INFO.
4. Empty/malformed output or missing required citations → INVALID, not an invented NEI answer.

A model can still decompose the claim incorrectly or assign a wrong part verdict. The aggregation guarantees consistency with those reported verdicts, not factual correctness.

Recommended test after confirmation: all ten original pilot claims, reusing their exact saved RAG evidence and excerpt IDs. No new retrieval, article fetching, embeddings or temporary databases are necessary for a prompt-only replay. Preserve the original outputs and write new outputs separately. The model, evidence, seed, temperature, context and original output-token cap remain fixed for the first replay; incomplete/length-limited outputs count as INVALID and are reported. No silent retries or tuning on individual claims.

The six previously wrong RAG IDs are 4807, 219203, 178210, 47172, 118664 and 61564. Testing only those would be targeted debugging, not a fair overall improvement score. Because these ten examples have already been inspected, even an all-ten replay is development analysis; a later untouched sample is needed for a stronger claim.

The 19K evaluation remains stopped. Await confirmation before implementing and running this revision.
