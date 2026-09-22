## Hva gjør denne endringen?

<!-- Kort beskrivelse. Hvorfor, ikke bare hva. -->

## Sjekkliste

- [ ] `python -m pytest -q` passerer
- [ ] `node --test web/test/*.mjs` passerer
- [ ] `ruff check flychess tests` er rent
- [ ] `docs/SPEC.md` er oppdatert hvis jeg har endret noe som står der
- [ ] `CHANGELOG.md` er oppdatert hvis brukeren merker endringen

## Motor og paritet

Berører denne endringen motoren?

- [ ] Nei — bare grensesnitt, dokumentasjon eller verktøy
- [ ] Ja, og paritetstesten passerer uten at toleransen er justert

<!--
Er svaret «ja» på den andre: forklar kort hvorfor endringen var nødvendig.
Paritetstesten er prosjektets viktigste løfte — at flua spiller nøyaktig det samme i nettleseren
som i Python. Hvis den må endres for at endringen skal gå gjennom, er det som regel endringen det
er noe galt med, ikke testen.
-->

## Skjermbilde

<!-- Obligatorisk for visuelle endringer: legg ved før og etter. -->

| Før | Etter |
|---|---|
|  |  |

## Språk

- [ ] All brukertekst jeg har lagt til er på norsk, ligger i `web/i18n.js` eller `web/slik.js`, og
      bruker æ, ø og å der det skal
- [ ] Motorkoden og testene for den er fortsatt på engelsk
