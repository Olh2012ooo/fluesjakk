# Fluesjakk

**Sjakk mot en fluehjerne.**

Motstanderen din er ikke en sjakkmotor. Det er et nevralt nettverk med **134 209 nevroner og
2 700 513 koblinger**, koblet nøyaktig slik hjernen til en fruktflue er koblet — hentet fra
[FlyWire](https://flywire.ai)-rekonstruksjonen av en voksen *Drosophila melanogaster*. Hver kobling
i modellen finnes i den ekte flua. Sjakkbrettet sendes inn i fluas sanseceller, aktiviteten i de
nedstigende og motoriske nevronene leses ut som et sjakktrekk, og ingenting annet velger trekkene.

Spillet finnes i tre styrker — **larve**, **flue** og **superflue** — og kjører helt lokalt på din
egen datamaskin.

> **Status:** fungerende spill og komplett verktøykjede. Flua er en svak sjakkspiller (se
> [Hvor sterk er den?](#hvor-sterk-er-den)) — poenget er at det virkelig er flua som spiller.

**Laget og videreutviklet av Oliver Lysø Hellevik.** Dette er en norsk videreutvikling av prosjektet
[fly-chess](https://github.com/cesp99/fly-chess). Hva som er arvet og hva som er nytt står i
[CREDITS.md](CREDITS.md).

---

## Innhold

- [Hva er Fluesjakk?](#hva-er-fluesjakk)
- [Spill mot en flue](#spill-mot-en-flue)
- [Slik fungerer det](#slik-fungerer-det)
- [Hvordan flua tenker](#hvordan-flua-tenker)
- [Slik spiller du](#slik-spiller-du)
- [Vanskelighetsgrader](#vanskelighetsgrader)
- [Teknologien](#teknologien)
- [Modell](#modell)
- [Personvern](#personvern)
- [Systemkrav](#systemkrav)
- [Installasjon](#installasjon)
- [Utvikling](#utvikling)
- [Testing](#testing)
- [Lisens](#lisens)
- [Credits](#credits)

---

## Hva er Fluesjakk?

Sjakk er et spill med fullstendig informasjon og et enormt antall muligheter. En fruktflue har ingen
som helst formening om sjakk. Fluesjakk setter de to tingene sammen ved å la et stykke ekte
nevrobiologi — koblingskartet over en hel fluehjerne — være det som velger trekkene.

Resultatet er et spill som er lett å komme i gang med, vanskelig å slutte å se på, og som samtidig
er ærlig om hva det er: et eksperiment som er gjort spillbart, ikke en sjakkmotor med et
fluekostyme.

Du kan spille mot den i nettleseren eller som en vanlig Windows-app. Alt regnes ut på din egen
maskin. Det finnes ingen konto, ingen sky og ingen sporing, og etter at modellen er lastet ned én
gang, virker spillet uten internett.

---

## Spill mot en flue

Flua ser brettet gjennom 2 048 sanseceller — og, i den nyeste modellen, gjennom **5 543
fotoreseptorer** i to fasettøyne. Den «tenker» i 16 gjentakelser gjennom 134 209 nevroner, og
resultatet leses ut som en sannsynlighet for hvert av de 4 168 lovlige trekkene pluss en gjetning på
hvem som vinner.

Underveis kan du se hva som skjer: nevronaktiviteten tegnes opp der nevronene faktisk sitter i
fluehjernen, fotoreseptorene farges etter hvilke ruter de ser på, og etter hvert trekk kan du spille
av tanken steg for steg og se signalet bevege seg fra øynene gjennom synslappene og inn i
sentralhjernen.

Det er ikke animasjon som er laget for å se bra ut. Det er de faktiske tallene nettverket regnet ut,
tegnet opp mens de skjer.

---

## Slik fungerer det

Dette er den delen det er verdt å lese hvis du vil forstå hva du egentlig spiller mot. En grundigere
og mer pedagogisk versjon ligger i spillet selv, under **Slik fungerer det**.

### Connectomet

FlyWire er en rekonstruksjon av en hel insekthjerne ved synapsenivå, laget ved å skjære hjernen i
tynne skiver, fotografere hver skive med elektronmikroskop og spore hver nervecelle og hvert
koblingspunkt gjennom bildene. Den offentlige utgaven (Codex snapshot 783) inneholder rundt 139 000
nevroner og titalls millioner synapser, med celletype, posisjon i hjernen og signaltype for de
fleste cellene.

Fluesjakk bruker dette kartet direkte som koblingsmønster i et nevralt nettverk:

| | |
|---|---|
| Nevroner i modellen | **134 209** (de med minst én kobling på ≥ 5 synapser) |
| Koblinger | **2 700 513** nevron-til-nevron-par |
| Synapser bak koblingene | **34 153 566** |
| Sanseceller brettet sprøytes inn i | 2 048 |
| Nedstigende og motoriske nevroner trekket leses ut av | 1 415 |
| Fotoreseptorer (nyeste modell) | 5 543 |
| Gjentakelser per trekk | 16 |
| Antall mulige trekk | 4 168 |

### Fra brett til tall

Et nevralt nettverk kan ikke se et sjakkbrett. Stillingen kodes derfor som **20 plan** på 8 × 8
ruter: ett per brikketype og farge, ett for hvilken side som skal trekke, plan for rokaderettigheter
og plan for en passant. Alt regnes fra **den som skal trekke sitt synspunkt**, så stillingen snus
hver gang det er den andre sin tur. Det er derfor den samme hjernen kan spille både hvit og svart.

### Gjennom nettverket

For hvert tidssteg regner hvert nevron ut en ny verdi:

```
h₀ = 0
for t in 1..16:
    pre      = W · hₜ₋₁ + bias              W: connectomet, sparsomt (2,7 mill. koblinger)
    pre[inn] += W_inn · brett + b_inn       brettet sprøytes inn i 2 048 sanseceller
    hₜ       = (1 − a) ⊙ hₜ₋₁ + a ⊙ relu(pre)      a = hvor fort nevronet følger signalet
policy = softmax(P · h₁₆[ut])               1 415 nevroner → 4 168 trekk-sannsynligheter
verdi  = tanh(V · h₁₆[ut])                  … → hvor godt flua tror den står
```

`(1 − a) ⊙ hₜ₋₁`-leddet gjør at litt av den forrige aktiviteten henger igjen. Det er derfor signalet
faktisk rekker å spre seg gjennom hjernen i stedet for å dø ut med én gang.

### Hva som er biologisk fastsatt, og hva som er trent opp

Dette er den viktigste nyanseringen i hele prosjektet:

| Del | Kilde | Trent? |
|---|---|---|
| Hvilke koblinger som finnes | connectomet | **nei — aldri** |
| Fortegnet til hver kobling (stimulerende/hemmende) | nevrotransmitter fra bildene | **nei — aldri** |
| Styrken på hver kobling | utgangspunkt i antall synapser | ja |
| Kvileverdi og tidskonstant per nevron | — | ja |
| Hvordan brettet vises til flua | — | ja |
| Hvordan trekkene leses ut | — | ja |

Fortegnene følger Dale's lov: acetylkolin virker stimulerende, GABA og glutamat hemmende. Det er
rundt 11,9 millioner tall som er justert under treningen, hvorav 2,7 millioner er styrken på
koblinger som finnes fra før og aldri kan bytte fortegn.

**Koblingskartet er fluas. Hvordan det brukes, er trent opp.** Den delen som ligner minst på en
flue, er hvordan brettet presenteres — en flue har ikke noe sjakkbrett-organ, og det er nettopp her
mesteparten av «sjakken» i modellen sitter.

### Hvordan den ble trent

1. **Etterligning.** Modellen så rundt 57 millioner stillinger fra omtrent 780 000 menneskelige
   partier fra Lichess, der begge spillere var ratet 1800 eller høyere, og ble trent til å gjette
   hvilket trekk mennesket spilte.
2. **Selvspill.** Deretter spilte den mot seg selv med Monte-Carlo-tresøk og lærte av resultatene.
   Nye kandidater ble bare godtatt hvis de slo den forrige beste i en kontrollert match.

Modellen som følger med spillet, er den nyeste av dem (`fly3`), trent i 264 000 steg.

---

## Hvordan flua tenker

Når flua skal velge et trekk, skjer dette:

1. **Brettet kodes** til 20 plan på 64 ruter, fra den som skal trekke sitt synspunkt.
2. **Sansecellene får signalet.** Hver av de 2 048 cellene regner ut en vektet sum av brettet.
3. **Nettverket kjører 16 runder.** Hvert nevron summerer signalene sine — men bare fra de nevronene
   det faktisk er koblet til i fluas hjerne — og oppdaterer verdien sin.
4. **Utlesingen skjer til slutt.** Aktiviteten i de 1 415 nedstigende og motoriske nevronene leses ut
   som en **policy** (hvor sannsynlig hvert trekk er) og en **verdi** (ett tall mellom −1 og 1 for
   hvor godt flua tror den står).
5. **Ulovlige trekk settes til null** og resten skaleres opp, så flua aldri kan spille et ulovlig
   trekk.
6. **Ett trekk velges.** Hvordan, avhenger av vanskelighetsgraden.

Verdien gjelder alltid den som skal trekke. Når flua er i trekket, er verdien altså fluas egen tro på
stillingen — det er den som driver humøret i panelet ved brettet, og den er ikke en
sjakkmotorvurdering.

Den nyeste modellen har også **øyne**: 5 543 fotoreseptorer fra fluas to fasettøyne ser hver sin rute
på brettet og sender signalet gjennom de ekte synsbanene — lamina, medulla og lobula — før
sentralhjernen får høre om det. Panelet **Fluas øyne** viser dette direkte.

---

## Slik spiller du

1. **Last ned spillet.** Windows: `Fluesjakk-x.y.z-Setup.exe` fra
   [Releases](https://github.com/oliverhellevik/fluesjakk/releases). Vil du ikke installere, kan du
   bruke zip-utgaven og pakke den ut hvor du vil.
2. **Åpne det.** Dobbeltklikk på `Fluesjakk.exe`. Ingen terminal, ingen Python, ingen Node.
3. **Første gang:** hvis modellen ikke allerede følger med, spør spillet om å laste den ned
   (omtrent 50 MB). Det gjøres én gang. Deretter virker spillet uten internett.
4. **Velg side.** Hvit, svart eller tilfeldig.
5. **Velg vanskelighetsgrad.** Larve, flue eller superflue.
6. **Spill.** Dra brikkene med musa eller fingeren, eller bruk piltastene og Enter.

Under partiet kan du snu brettet med **F**, angre med **Ctrl+Z**, begynne på nytt med **N**, se på
fluas tanke med avspillingsknappen under brettet, og se hva flua faktisk ser i øypanelet.

---

## Vanskelighetsgrader

Alle tre bruker nøyaktig samme hjerne. Det som endres, er hvor mye arbeid som legges i å velge blant
trekkene.

### 🐛 Larve

Flua trekker et tilfeldig trekk fra policyen sin, med høy temperatur. Selv trekk den bare er 3 %
sikker på, kan bli spilt. Resultatet er uforutsigbart og svakt — men det er fortsatt fluas egne
sannsynligheter som avgjør, ikke en terning alene.

### 🪰 Flue

Flua tar de tre trekkene den liker best, spiller hvert av dem på brettet og spør verdihodet sitt:
«hvordan liker motstanderen dette?» Trekket motstanderen liker dårligst, blir spilt. Ett steg med
framsyn — nok til å unngå de verste tabbene, ikke nok til å regne ut en kombinasjon.

### 🧠 Superflue

Nå får flua tenke skikkelig. Den kjører et **Monte-Carlo-tresøk**: den undersøker stillingen gren for
gren, og hver gang den trenger en vurdering av en ny stilling, er det fluehjernen som gir den — både
hvilke trekk som virker lovende og hvor god stillingen er.

Antall gjennomganger avhenger av maskinen din: rundt 400 hvis nettleseren får bruke skjermkortet,
rundt 40 i ren JavaScript. Det er den samme tanken, bare langsommere.

---

## Teknologien

```
FlyWire-tabeller ──► fly build-brain ──► BrainGraph (sparsom matrise)        data/brain/full.npz
Lichess .pgn.zst ──► fly build-shards ─► stillinger (plan, trekk, resultat) data/shards/*.npz
                                                    │
                                                    ▼
             ┌────────────────────────── fly train ──────────────────────────┐
             │  steg 1: etterligning     steg 2: selvspill med MCTS          │
             │  FlyBrain: brett ──► W_inn ──► [ 134k nevroner, 2,7M koblinger ]×16
             │                                    └──► policy (4168) / verdi (1)│
             └──────────────► runs/<kjøring>/{ckpt-N.pt, latest.pt, metrics} ─┘
                                  │                          │
                  fly dashboard ◄─┘                          └─► fly export-web ──► web/model/brain.flyb
                                                                                        │
                                                    nettleser: loader.js → flybrain.js → mcts.js
                                                               (i en Web Worker)
                                                    paritet: tests/vectors/model.json == web/test/parity.test.mjs
```

### Delene

| Mappe | Hva den gjør |
|---|---|
| `flychess/connectome/` | Laster ned og tolker FlyWire-tabellene, bygger den sparsomme hjernen |
| `flychess/chessenv/` | Brett → plan, trekk ↔ indeks, sjakkmiljøet |
| `flychess/model/` | `FlyBrain` — det rekurrente nettverket og den sparsomme mattemultiplikasjonen |
| `flychess/train/` | Etterligning, selvspill, Monte-Carlo-tresøk, treneren |
| `flychess/export/` | Eksport til nettleserformatet og referanseimplementasjonen i numpy |
| `flychess/play/` | Motoren med de tre vanskelighetsgradene, terminalgrensesnitt og lokal tjener |
| `web/` | Nettleserversjonen: grensesnitt, Web Worker-motor, visninger |
| `desktop/` | Skrivebordsappen for Windows |
| `docs/SPEC.md` | Kontrakten alle modulene følger |

### Nettleseren og Python regner det samme

Motoren finnes i to utgaver: én i PyTorch for trening og eksport, og én håndskrevet i JavaScript som
kjører i nettleseren. At de to faktisk er enige, er ikke noe vi tror på — det er testet.

Eksporten skriver også ut fasiten: tolv stillinger med logits og verdier regnet ut i Python, rett fra
de eksporterte bytene. Testen `web/test/parity.test.mjs` laster den samme modellen med nettleserens
egen kode og krever at hvert eneste tall stemmer innenfor 1e-2. I praksis er forskjellen rundt 1e-4.

Det er derfor grensesnittet kan redesignes fritt: så lenge motoren og eksporten står urørt, spiller
flua nøyaktig de samme trekkene.

---

## Modell

Modellen er på omtrent **60 MB rå** og **50 MB komprimert**.

### Hvorfor den ikke ligger i kodelageret

Modellvektene er avledet av FlyWire-connectomet og er derfor lisensiert under **CC BY-NC 4.0** —
ikke under MIT slik koden er. De kan brukes fritt til ikke-kommersielle formål med attribusjon, men
de kan ikke selges eller brukes kommersielt.

Fordi vektene og koden har ulike lisenser, holder vi dem atskilt. Kodelageret inneholder bare kode.
Vektene hentes på én av to måter:

**1. Spillet laster dem ned selv, første gang.** Med framdriftsvisning, og filene kontrolleres mot
sjekksummen i headeren før de tas i bruk. Deretter ligger de i hurtiglageret og spillet virker
uten internett. Dette er standardveien for nettsideversjonen.

**2. De legges ved siden av spillet.** Legg `brain.json` og `brain.flyb.gz` i `web/model/` før du
bygger, så pakkes de inn og spillet trenger aldri nettet. Dette er slik den ferdige
Windows-utgaven bygges.

Kilden er modellarkivet på Hugging Face:
<https://huggingface.co/cesp99/fly-chess> (`web/brain.json`, `web/brain.flyb.gz`).

### Hvor sterk er den?

Etterligningstreet på 5 120 menneskelige stillinger som ikke var en del av treningen:

| Modell | Topp-1 | Topp-3 |
|---|---|---|
| `fly1` — 8 tidssteg, brett via 2 048 sanseceller | 34,1 % | 58,4 % |
| `fly2` — 16 tidssteg, mer data, fp32 | 33,1 % | 57,2 % |
| **`fly3`** — som fly2, pluss øyne, homeostatiske gain, flertrinns utlesing og selvspill | **37,6 %** | **63,4 %** |

Uten søk er flua en overkommelig motstander som spiller fornuftige åpninger og posisjonssjakk, men
som fortsatt overser taktikk. Med søk (superflue) blir det en reell kamp. Den taper for enhver
ordentlig sjakkmotor, og det er meningen.

---

## Personvern

Fluesjakk kjører **helt og holdent på din egen datamaskin**.

- Ingen konto, ingen innlogging.
- Ingen data sendes til noen server. Det finnes ingen telemetri og ingen analyseverktøy.
- Den eneste nettverksforespørselen spillet noensinne gjør, er å hente modellfilen første gang.
  Har du den lokalt, gjør spillet ingen forespørsler i det hele tatt.
- Topplisten og innstillingene lagres i nettleserens lokale lager på din maskin, og forlater den
  aldri.

Du kan bekrefte dette selv: åpne nettverksfanen i utviklerverktøyene og se at det ikke skjer noe
etter at modellen er lastet.

---

## Systemkrav

**Windows:** Windows 10 eller nyere, 64-bit. Ingen installasjon av Python, Node eller andre
utviklerverktøy. Omtrent 250 MB ledig plass.

**Nettleser:** en moderne nettleser med støtte for ES-moduler og Web Workers — Chrome, Edge, Firefox
eller Safari i nyere versjon. Har nettleseren WebGPU, kjører nettverket på skjermkortet og flua
svarer merkbart raskere.

**Ytelse:** i ren JavaScript tar ett framoverpass rundt 90 ms på en vanlig laptop, så et trekk på
flue-nivå svarer på under et sekund. Superflue gjør mange gjennomganger og bruker derfor flere
sekunder per trekk uten skjermkort. Med WebGPU går det rundt 25 ganger raskere.

---

## Installasjon

### For deg som bare vil spille

1. Gå til [Releases](https://github.com/oliverhellevik/fluesjakk/releases).
2. Last ned `Fluesjakk-x.y.z-Setup.exe`.
3. Kjør den og følg veiviseren.
4. Åpne **Fluesjakk** fra startmenyen.

Installasjonsfilen er ikke kodesignert (det koster penger og prosjektet er gratis), så Windows kan
vise en advarsel fra SmartScreen første gang. Velg **Mer informasjon** → **Kjør likevel**.

Vil du ikke installere noe, kan du laste ned zip-utgaven i stedet og pakke den ut hvor du vil.

### For deg som vil spille i nettleseren

```bash
git clone https://github.com/oliverhellevik/fluesjakk && cd fluesjakk
python -m http.server 8000 --directory web
# åpne http://localhost:8000
```

Spillet laster ned modellen selv ved første besøk.

> Nettleseren nekter å laste moduler og arbeidertråder fra `file://`. Derfor må siden kjøres fra en
> tjener — den innebygde `python -m http.server` holder i massevis.

---

## Utvikling

```bash
git clone https://github.com/oliverhellevik/fluesjakk && cd fluesjakk
uv sync                       # eller: pip install -e . && pip install --group dev
source .venv/bin/activate
```

### Hente modellen

```bash
# hele hjernen fra Hugging Face
hf download cesp99/fly-chess --include "web/brain.*" --local-dir modell
mkdir -p web/model && cp modell/web/* web/model/
```

### Bygge skrivebordsappen

```bash
cd desktop
npm install
npm start                     # kjører appen direkte
npm run build                 # Windows-installer + zip i dist/
```

Har du ikke utviklermodus i Windows, kan bygget stoppe på at electron-builder ikke får lage
symbolske lenker. Legg da til `--config.win.signAndEditExecutable=false`. På GitHub Actions, som
kjører med de rettighetene som trengs, bygges den fulle utgaven.

### Hele verktøykjeden

```bash
fly download                  # FlyWire-tabellene og Lichess-partier
fly build-brain               # bygger hjernen fra tabellene
fly build-shards --months 2014-01
fly train --run fly1
fly export-web --run fly1     # web/model/ + tests/vectors/model.json
fly play --run fly1           # terminal
fly play --run fly1 --gui     # lokal nettside
fly dashboard --run fly1      # sanntidsdashboard
```

Se [docs/SPEC.md](docs/SPEC.md) for kontrakten mellom modulene og
[web/README.md](web/README.md) for nettlesersiden i detalj.

---

## Testing

```bash
# Python: motor, trening, eksport, koding
python -m pytest -q

# Nettleser: motor, paritet mot Python, grensesnittet i headless Chromium
node --test web/test/*.mjs

# Stil
ruff check flychess tests
```

Paritetstesten krever `tests/vectors/model.json`, som genereres fra den eksporterte modellen:

```bash
fly export-web --run fly1                       # fra en trent kjøring
python -c "from flychess.export.testvectors import write_model_vectors; write_model_vectors(model_dir='web/model')"
```

Uten den hopper paritetstesten over med en melding i stedet for å feile.

---

## Lisens

**Koden** i dette prosjektet er utgitt under [MIT-lisensen](LICENSE), © 2026 Oliver Lysø Hellevik.
Deler av koden er arvet fra fly-chess og står også under MIT, © 2026 Carlo Esposito — den
opphavsretten er beholdt som lisensen krever.

**Modellvektene** er avledet av FlyWire-dataene og er underlagt **CC BY-NC 4.0**. De kan brukes fritt
til ikke-kommersielle formål med attribusjon, men ikke selges eller brukes kommersielt. Det betyr i
praksis at du gjerne må lese, endre og dele videre koden, men at spillet ikke kan selges så lenge det
bygger på FlyWire-connectomet.

**FlyWire-dataene** er CC BY-NC 4.0. **Lichess-partiene** er CC0. **chess.js** er BSD-2-Clause.
**python-chess** er GPL-3.0-or-later og brukes bare av Python-siden, som derfor ikke inngår i
Windows-utgaven.

Full oversikt i [CREDITS.md](CREDITS.md) og i [LICENSE](LICENSE).

---

## Credits

**Utvikler av denne versjonen:** Oliver Lysø Hellevik.

Connectomet, celletypene og nevrotransmitterne er ikke laget for dette prosjektet. De er
forskningsresultater fra FlyWire-konsortiet:

- Dorkenwald, S. *et al.* Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138 (2024).
  <https://doi.org/10.1038/s41586-024-07558-y>
- Schlegel, P. *et al.* Whole-brain annotation and multi-connectome cell typing of *Drosophila*.
  *Nature* 634, 139–152 (2024). <https://doi.org/10.1038/s41586-024-07686-5>

Og grunnlaget dette bygger på er [fly-chess](https://github.com/cesp99/fly-chess) av Carlo Esposito.

Se [CREDITS.md](CREDITS.md) for full attribusjon, inkludert hva som er arvet og hva som er nytt.
