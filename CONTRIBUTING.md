# Bidra til Fluesjakk

Tak for at du vil være med. Dette dokumentet sier hvordan prosjektet er organisert, hva som er
viktig å ikke ødelegge, og hvordan du får endringene dine inn.

---

## Det viktigste å vite

Fluesjakk er bygget rundt én regel: **flua skal spille det samme, uansett hva grensesnittet gjør.**

Motoren er pinnet mot Python av en paritetstest. Endrer du noe i `web/engine/`, må du regne med at
paritetstesten stopper deg hvis tallene endrer seg — og da er det motoren det er noe galt med, ikke
testen. Det samme gjelder `flychess/model/`, `flychess/chessenv/` og `flychess/export/`.

Grensesnittet derimot — `index.html`, `style.css`, `app.js`, `board.js`, `i18n.js`, `slik.js` — kan
endres så mye du vil. Så lenge du ikke tar i `web/engine/` eller `web/vendor/`, spiller flua
nøyaktig som før.

| Kan endres fritt | Må behandles varsomt |
|---|---|
| Grensesnitt, layout, farger, tekst | `web/engine/**` |
| `web/app.js`, `web/board.js`, `web/i18n.js` | `web/vendor/chess.js` |
| Innstillinger og nye visninger | `flychess/model/**`, `flychess/export/**` |
| Dokumentasjon | `flychess/chessenv/encoding.py`, `web/engine/encoding.js` |
| Tester for grensesnittet | `docs/SPEC.md` |

`docs/SPEC.md` er kontrakten alle modulene følger. Endrer du noe som står der, må du oppdatere
spesifikasjonen i samme draforespørsel.

---

## Språk

- **Brukergrensesnittet og all brukerdokumentasjon er på norsk bokmål.** Bruk æ, ø og å. Skriv
  naturlig norsk, ikke oversatt ord for ord, og følg norsk sjakktradisjon: *sjakk*, *sjakk matt*,
  *patt*, *remis*, *bondeforvandling*, *springer*, *løper*, *tårn*, *bonde*.
- **Motorkoden, testene for den og `docs/SPEC.md` er på engelsk**, i tråd med den arvede koden og
  spesifikasjonen. Ikke bland språkene i samme fil.
- All tekst brukeren ser, skal ligge i `web/i18n.js` eller `web/slik.js`. Testen
  «ingen engelske rester i grensesnittet» i `web/test/app.test.mjs` feiler hvis det dukker opp
  engelske ord i grensesnittet.

---

## Sette opp

```bash
git clone https://github.com/oliverhellevik/fluesjakk && cd fluesjakk
uv sync                     # eller: pip install -e . && pip install --group dev
source .venv/bin/activate

python scripts/hent_modell.py    # modellen til web/model/ (trengs for paritetstesten)
```

For skrivebordsappen i tillegg:

```bash
cd desktop && npm install
```

---

## Før du sender inn

Kjør alt dette og sørg for at det er grønt:

```bash
python -m pytest -q
node --test web/test/*.mjs
ruff check flychess tests
```

Har du endret grensesnittet, ta et skjermbilde før og etter. Det er den raskeste måten å få en
vurdering av om endringen faktisk ser bedre ut — og for visuelle endringer er det ofte hele
diskusjonen.

### Sjekkliste

- [ ] Alle testene over passerer.
- [ ] Har du endret motoren: paritetstesten passerer uten at toleransen er justert.
- [ ] Har du lagt til brukertekst: den er norsk, ligger i `i18n.js`, og har æ, ø og å der det skal.
- [ ] Har du lagt til en funksjon: den er testet, og testen feiler hvis funksjonen fjernes.
- [ ] Har du endret oppførsel brukeren merker: `CHANGELOG.md` er oppdatert.
- [ ] Ingenting i den arvede koden er endret uten at det står hvorfor i koden.

---

## Skrivestil i koden

- Kommentarer forklarer **hvorfor**, ikke hva koden gjør. Koden sier hva.
- Skriv kommentarer på samme språk som resten av filen.
- Ikke skriv om store filer bare for å rydde. Hver endring skal kunne leses og forstås alene.
- Nye avhengigheter skal begrunnes. Prosjektet har bevisst få, og nettlesersiden har ingen — den er
  statiske filer uten byggetrinn.

---

## Rapportere feil

Bruk skjemamalen for feil. Legg ved den **tekniske loggen**: åpne spillet, gå til **Om** →
**Feilsøking** → **Kopier teknisk logg**. Den inneholder nettleserversjon, modellinformasjon og de
siste hendelsene, uten personlige opplysninger.

Er du usikker på om noe er en feil: beskriv hva du gjorde, hva du forventet og hva som skjedde.
Det holder lenger enn en gjetning på årsaken.

---

## Lisens på bidrag

Ved å sende inn en endring godtar du at den gis ut under prosjektets lisens: MIT for kode, og at
modellvekter og avledede data forblir under CC BY-NC 4.0.

Bidra aldri med kode eller data du ikke har rett til å dele. Særlig: ikke legg modellvekter,
FlyWire-tabeller eller treningsdata inn i kodelageret — de har andre lisenser enn koden, og hentes
ned separat. Se [CREDITS.md](CREDITS.md).
