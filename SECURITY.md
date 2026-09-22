# Sikkerhet

## Hva Fluesjakk er, sikkerhetsmessig

Fluesjakk er et lokalt spill. Det har ingen server, ingen konto, ingen database og ingen
brukerinnhold fra andre. Det gjør angrepsflaten liten, men ikke tom.

**Spillet gjør nøyaktig én nettverksforespørsel:** den henter modellfilen fra Hugging Face første
gang. Etter det kjører alt lokalt. Har du modellen fra før, gjør spillet ingen forespørsler i det
hele tatt.

**Spillet samler ikke inn noe.** Ingen telemetri, ingen analyse, ingen krasjrapportering. Topplisten
og innstillingene ligger i nettleserens lokale lager og forlater aldri maskinen.

## Hvordan det er bygget for å tåle litt

- Skrivebordsappen kjører nettsiden i et sandkassevindu uten tilgang til Node eller filsystemet
  (`contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`).
- Den lokale tjeneren i skrivebordsappen lytter bare på `127.0.0.1` og avviser forespørsler som
  peker utenfor spillmappen.
- Den nedlastede modellen kontrolleres mot sjekksummen i headeren før den tas i bruk. Stemmer den
  ikke, blir filen forkastet og lastet ned på nytt.
- Nettlesersiden har ingen eksterne avhengigheter i det hele tatt — ingen CDN, ingen tredjeparts
  skript. Alt den laster, ligger i denne pakken.

## Ting det er verdt å vite

- **Installasjonsfilen er ikke kodesignert.** Det er et bevisst valg: et sertifikat koster penger, og
  prosjektet er gratis. Windows viser derfor en SmartScreen-advarsel første gang. Du kan bekrefte
  at filen er ekte ved å sammenligne sjekksummen med den som står i utgivelsen.
- **Modellen er data, ikke kode.** Den tolkes av `web/engine/loader.js`, som avviser blober med feil
  størrelse eller feil sjekksum. En modell fra en uvedkommende kilde er likevel ikke noe spillet kan
  garantere innholdet i — hent den fra prosjektets egen utgivelse eller fra
  [cesp99/fly-chess](https://huggingface.co/cesp99/fly-chess).

## Rapportere et sikkerhetsproblem

Bruk **ikke** et åpent issue for sikkerhetsproblemer.

Bruk i stedet GitHubs innebygde rapportering under **Security** → **Report a vulnerability** i
kodelageret. Da holdes saken privat mens den undersøkes.

Ta med:

- hva problemet er og hvilken del av prosjektet det gjelder
- hvordan det kan gjenskapes, så konkret som mulig
- hvilken versjon av Fluesjakk og hvilket operativsystem du bruker

Du får svar så snart det lar seg gjøre. Dette er et hobbyprosjekt uten et supportteam bak seg, så
beregn noe tid.

## Støttede versjoner

Sikkerhetsrettinger gis for den nyeste utgaven. Eldre utgaver får ingen oppdateringer.
