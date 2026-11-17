# Historical symbolic verifier audit

100/100 saved decisions and reasons reproduced offline. Rules, labels and old scores unchanged.

- 96 claims returned unsupported_claim_pattern before any Wikidata query: narrow rule coverage.
- Samuel Witwer search returned Sam Witwer with Samuel Witwer as an alias; inherited search_exact checks labels only.
- October, May and November expressions pass the temporal guard and reach P19 birthplace rules: a parsing defect.
- David Beckham search contained two exact labels; abstention reflects unresolved ambiguity, not an HTTP outage.
- The Charcot response already contained birth date P569 = 1825-11-29, but the parser checked P19 places instead.
- Halle Berry object_text retained Texas on August 14, 1966 and was compared with Cleveland label/aliases as one string.
- V2 inherits place matching from v1. These results audit this narrow implementation, not symbolic verification in general.
