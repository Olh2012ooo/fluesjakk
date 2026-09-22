# desktop/ — Fluesjakk som Windows-app

Pakker nettsiden i `web/` som et vanlig program. Brukeren skal slippe å installere Python, Node
eller noe annet, og skal ikke trenge en terminal.

## Hvorfor Electron

Alternativene ble vurdert opp mot det som faktisk betyr noe her:

| | Electron | Tauri |
|---|---|---|
| Størrelse | ~170 MB installert | ~10 MB |
| Krever | ingenting utover npm | Rust-verktøykjeden (~1,5 GB) |
| Rendering | egen Chromium | systemets WebView2 |
| WebGPU | følger Chromium | avhenger av WebView2-versjonen |
| Risiko | lav | høyere: bygget avhenger av en hel verktøykjede |

Tauri gir en mye mindre fil, og det er et reelt gode. Det som avgjorde saken, er at **WebGPU**
gjør superflue rundt 25 ganger raskere, og at Electron garanterer at motoren oppfører seg likt på
alle maskiner fordi den har sin egen nettleser med seg. Med Tauri ville ytelsen og til og med
funksjonaliteten avhengt av hvilken WebView2-versjon som tilfeldigvis lå på maskinen.

Størrelsen er prisen for at spillet virker likt overalt. Det er en bevisst avveining, ikke en
forglemmelse.

## Slik virker det

Appen starter en liten statisk tjener på `127.0.0.1` og åpner et vindu mot den. Grunnen til at den
ikke bare åpner `index.html` direkte: nettlesere nekter å laste ES-moduler, Web Workers og
Cache API fra `file://`. En tjener på lokalmaskinen løser det uten at noe går ut på nettet.

Tjeneren lytter bare på `127.0.0.1`, og avviser forespørsler som peker utenfor spillmappen.

Vinduet kjører i sandkasse: `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`.
Siden har ingen tilgang til filsystemet eller Node. Eksterne lenker åpnes i systemets nettleser i
stedet for inne i spillet.

## Bygge

```bash
npm install
npm start          # kjør direkte fra kildekoden
npm run build      # Windows-installer + zip i ../dist/
```

Vil du at modellen skal følge med, slik at spillet virker helt uten internett fra første stund:

```bash
python ../scripts/hent_modell.py    # legger brain.json og brain.flyb.gz i ../web/model/
npm run build
```

Er `web/model/` tom, lastes modellen ned av spillet selv ved første oppstart i stedet.

### Symbolske lenker på Windows

electron-builder pakker ut et verktøysett som inneholder symbolske lenker. På Windows krever det
enten utviklermodus eller administratorrettigheter. Uten det stopper bygget med
«Cannot create symbolic link».

To måter å komme rundt det:

- Slå på **Utviklermodus** i Windows-innstillingene (Innstillinger → System → Utviklere), eller
- kjør bygget uten å skrive inn app-ikon og versjonsinformasjon i .exe-filen:

  ```bash
  npx electron-builder --win --x64 --config.win.signAndEditExecutable=false
  ```

GitHub Actions kjører med de rettighetene som trengs, så utgivelsene bygges alltid i full utgave.

## Røyktest

For å sjekke at en pakket utgave faktisk starter (brukes av CI og av den som vil verifisere et
bygg):

```bash
FLUESJAKK_SMOKE=1 ./dist/win-unpacked/Fluesjakk.exe
```

Den skriver én linje med JSON til stdout og avslutter med 0 bare hvis siden kom opp og modellen
(lastet eller tilbudt nedlastet) er klar. Uten miljøvariabelen gjør den ingenting.

## Kodesignering

Utgaven er ikke kodesignert. Et sertifikat koster penger, og prosjektet er gratis. Windows viser
derfor en SmartScreen-advarsel første gang. Sjekksummene i utgivelsen lar deg bekrefte at filen er
den den utgir seg for å være.
