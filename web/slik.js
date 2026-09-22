// slik.js — innholdet på «Slik fungerer det»-siden og i feilsøkingsdialogen.
//
// Teksten er skrevet for en vanlig person som er nysgjerrig, ikke for en forsker. Den starter
// enkelt og går gradvis dypere. Alt som står her er etterprøvbart i koden: ingen påstander om
// fluas indre liv som ikke følger av det nettverket faktisk regner ut.

const FLYT = [
  ['Sjakkbrettet', 'Stillingen du ser.'],
  ['20 plan', 'Brettet gjøres om til 20 lag med 64 ruter hver, sett fra den som skal trekke.'],
  ['Sansecellene', '2 048 sanseceller får hver sin vektede sum av de 20 planene.'],
  ['Nevronnettet', '134 209 nevroner og 2,7 millioner koblinger, kjørt 16 ganger etter hverandre.'],
  ['Nevronaktiviteten', 'Etter siste runde lyser de nedstigende og motoriske nevronene.'],
  ['Policy og verdi', 'Aktiviteten leses ut som én sannsynlighet per trekk og én dom over stillingen.'],
  ['Mulige trekk', 'Bare lovlige trekk beholdes — resten settes til null.'],
  ['Fluas trekk', 'Ett trekk velges og spilles.'],
];

export const SLIK_INNHOLD = `
<div class="slik">
  <header class="slik-topp">
    <p class="eyebrow">Slik fungerer det</p>
    <h1>Slik fungerer Fluesjakk</h1>
    <p class="lede">
      Fluesjakk prøver ikke å etterligne hvordan en flue tenker. Sjakkbrettet blir i stedet sendt inn
      i en modell basert på koblingene i en faktisk rekonstruert fruktfluehjerne, og det som kommer
      ut i andre enden er et sjakktrekk.
    </p>
  </header>

  <section class="slik-seksjon">
    <h2>Hele veien, i åtte steg</h2>
    <ol class="flyt">
      ${FLYT.map(([t, f]) => `<li><span class="flyt-tittel">${t}</span><span class="flyt-forklaring">${f}</span></li>`).join('')}
    </ol>
    <p class="dim">Hvert eneste trekk flua gjør, går gjennom hele denne kjeden. Det finnes ingen
    reserveplan, ingen åpningsbok og ingen tilfeldighetsmotor som tar over når nettverket er uenig
    med seg selv.</p>
  </section>

  <section class="slik-seksjon">
    <h2>Hva er et connectome?</h2>
    <p>Et connectome er et kart over hvem som snakker med hvem i en hjerne. For fruktflua er dette
    kartet laget ved å skjære hjernen i tynne skiver, fotografere hver skive med elektronmikroskop og
    deretter spore hver eneste nervecelle og hvert eneste koblingspunkt gjennom bildene. Prosjektet
    heter <a href="https://flywire.ai" rel="noopener">FlyWire</a>, og resultatet er den største
    komplette hjernekartleggingen som finnes: rundt 139 000 nevroner og titalls millioner synapser
    hos en voksen fruktflue.</p>

    <p>Fluesjakk bruker dette kartet direkte. Hver kobling i modellen finnes i den ekte flua — det er
    ingen kobling som er lagt til fordi den ville vært praktisk for sjakk. Terskelen er fem synapser
    mellom et par nevroner, som i den offentlige utgivelsen.</p>

    <div class="bokser">
      <div class="boks"><b>134 209</b><span>nevroner i modellen</span></div>
      <div class="boks"><b>2 700 513</b><span>koblinger mellom dem</span></div>
      <div class="boks"><b>34 153 566</b><span>synapser bak koblingene</span></div>
      <div class="boks"><b>16</b><span>gjentakelser per trekk</span></div>
    </div>
  </section>

  <section class="slik-seksjon">
    <h2>Hvordan brettet blir til tall</h2>
    <p>Et nevronnettverk kan ikke se et sjakkbrett. Brettet må bli tall først. Stillingen deles opp i
    <b>20 plan</b>, der hvert plan er et rutenett på 8 × 8:</p>
    <ul class="punktliste">
      <li>ett plan per brikketype og farge — seks hvite og seks svarte</li>
      <li>ett plan for hvilken side som skal trekke</li>
      <li>plan for rokaderettigheter i de fire hjørnene</li>
      <li>plan for om en bonde kan slå en passant</li>
    </ul>
    <p>Alt regnes fra <b>den som skal trekke sitt synspunkt</b>. Det er ikke en detalj: stillingen
    snus altså hver gang det er den andre sin tur, slik at modellen alltid ser brettet «oppover».
    Det er derfor flua kan spille både hvit og svart med samme hjerne.</p>
  </section>

  <section class="slik-seksjon">
    <h2>Øynene — for modeller som har dem</h2>
    <p>Den nyeste modellen har i tillegg fått øyne. 5 543 fotoreseptorer fra fluas to fasettøyne
    er koblet til hver sin rute på brettet: venstre øye ser den ene halvdelen av brettet, høyre øye
    den andre. Hver fotoreseptor tar en liten vektet sum av de 20 planene for ruten den ser på, og
    sender resultatet videre inn i de ekte synsbanene — lamina, medulla og lobula — før sentralhjernen
    får høre noe som helst.</p>
    <p>Panelet <b>Fluas øyne</b> viser dette direkte. I visningen <i>ser</i> er hver fotoreseptor
    farget etter ruten den følger med på. I visningen <i>føler</i> er den farget etter hvor sterk
    inngangsstrømmen faktisk er i den stillingen flua sist så på. Ingen av fargene er pynt — de
    kommer fra tallene nettverket regnet ut.</p>
  </section>

  <section class="slik-seksjon">
    <h2>Hvordan et trekk blir til</h2>
    <p>Når brettet er inne i nettverket, skjer følgende, og dette er hele mekanismen:</p>
    <ol class="punktliste nummerert">
      <li>Brettet sprøytes inn i 2 048 sanseceller, ett tall per celle.</li>
      <li>Hver av de 134 209 nevronene regner ut en ny verdi ut fra summen av signalene den får,
          men bare fra de nevronene den faktisk er koblet til i fluas hjerne.</li>
      <li>Hver kobling har et <b>fortegn</b> som aldri endres: acetylkolin virker stimulerende,
          GABA og glutamat hemmende. Det er fluas egen biologi, ikke noe modellen har funnet på.</li>
      <li>Hele runden gjentas <b>16 ganger</b>. Mellom hver runde henger litt av den forrige
          aktiviteten igjen, slik at signalet faktisk rekker å spre seg gjennom hjernen.</li>
      <li>Etter siste runde leses aktiviteten til de <b>1 415 nedstigende og motoriske nevronene</b> av.</li>
      <li>Den leses ut på to måter: som en <b>policy</b> — hvor sannsynlig hvert av de 4 168 mulige
          trekkene er — og som en <b>verdi</b> — ett tall mellom −1 og 1 for hvor godt flua tror den står.</li>
      <li>Trekk som ikke er lovlige i stillingen settes til null, og resten skaleres opp.</li>
      <li>Ett trekk velges. Hvilket, avhenger av vanskelighetsgraden.</li>
    </ol>
    <p class="dim">Verdien gjelder alltid den som skal trekke. Når flua selv er i trekket, er verdien
    altså fluas egen tro på stillingen — det er den som driver humøret i panelet ved brettet.</p>
  </section>

  <section class="slik-seksjon">
    <h2>Hva er faktisk lært opp, og hva er biologisk fastsatt?</h2>
    <p>Dette er den viktigste nyanseringen i hele prosjektet:</p>
    <div class="tabell-wrap">
      <table class="slik-tabell">
        <tr><th>Del</th><th>Kilde</th><th>Trent?</th></tr>
        <tr><td>Hvilke koblinger som finnes</td><td>connectomet</td><td class="nei">nei — aldri</td></tr>
        <tr><td>Fortegnet til hver kobling</td><td>nevrotransmitter fra bilder</td><td class="nei">nei — aldri</td></tr>
        <tr><td>Styrken på hver kobling</td><td>utgangspunkt i antall synapser</td><td class="ja">ja</td></tr>
        <tr><td>Kvileverdi og tidskonstant per nevron</td><td>—</td><td class="ja">ja</td></tr>
        <tr><td>Hvordan brettet vises til flua</td><td>—</td><td class="ja">ja</td></tr>
        <tr><td>Hvordan trekkene leses ut</td><td>—</td><td class="ja">ja</td></tr>
      </table>
    </div>
    <p>Med andre ord: <b>koblingskartet er fluas, men hvordan det brukes er trent opp.</b>
    Det er rundt 11,9 millioner tall som er justert under treningen, hvorav 2,7 millioner er
    styrken på koblinger som finnes fra før og aldri kan bytte fortegn.</p>
    <p>Den delen som ligner minst på en flue, er hvordan brettet presenteres. En flue har ikke noe
    sjakkbrett-organ. Det er nettopp her mesteparten av «sjakken» i modellen sitter.</p>
  </section>

  <section class="slik-seksjon">
    <h2>Vanskelighetsgradene</h2>
    <p>Alle tre bruker nøyaktig samme hjerne. Det som endres er hvor mye arbeid som legges i å velge
    blant trekkene.</p>

    <h3>Larve</h3>
    <p>Flua trekker et tilfeldig trekk fra policyen sin, med høy temperatur. Det betyr at selv trekk
    flua bare er 3 % sikker på kan bli spilt. Resultatet er uforutsigbart og svakt — men det er
    fortsatt fluas egne sannsynligheter som avgjør, ikke en terning alene.</p>

    <h3>Flue</h3>
    <p>Flua tar de tre trekkene den liker best, spiller hvert av dem på brettet og spør verdihodet
    sitt: «hvordan liker motstanderen dette?» Trekket motstanderen liker dårligst, blir spilt.
    Det er ett steg med framsyn — nok til å unngå de verste tabbene, men ikke nok til å regne ut en
    kombinasjon.</p>

    <h3>Superflue</h3>
    <p>Nå får flua tenke skikkelig. Den kjører et <b>Monte-Carlo-tresøk</b>: den undersøker stillingen
    gren for gren, og hver eneste gang den trenger en vurdering av en ny stilling, er det fluehjernen
    som gir den — både hvilke trekk som virker lovende og hvor god stillingen er.</p>
    <p>Antall gjennomganger avhenger av maskinen din: rundt 400 på et skjermkort, rundt 40 i ren
    JavaScript. Det er den samme tanken, bare langsommere. Søket bruker en balansering som heter PUCT
    til å avgjøre hvilke grener som er verdt å se nærmere på.</p>
    <p class="dim">Underveis viser panelet hvor mange gjennomganger som er unnagjort. «Instinkt»-linjen
    i kommentaren er hva policyen alene ville spilt; noen ganger er søket enig, noen ganger
    overprøver det magefølelsen.</p>
  </section>

  <section class="slik-seksjon">
    <h2>Hva er forskjellen på Python-versjonen og nettleserversjonen?</h2>
    <p>Ingenting, regnemessig. Den samme modellen er eksportert til et kompakt binærformat, og
    nettleseren kjører nøyaktig de samme matteoperasjonene i en <b>Web Worker</b> — en egen tråd, slik
    at grensesnittet ikke fryser mens flua tenker. Har nettleseren WebGPU, går regnestykkene til
    skjermkortet i stedet for til prosessoren; er WebGPU utilgjengelig, kjøres alt i ren JavaScript
    uten at resultatet endrer seg.</p>
    <p>At de to faktisk er enige, er ikke noe vi tror — det er testet. Eksporten skriver også ut
    fasiten: tolv stillinger med logits og verdier regnet ut i Python. Testen
    <code>web/test/parity.test.mjs</code> laster den samme modellen med nettleserens egen kode og
    krever at hvert eneste tall stemmer innenfor 1e-2. I praksis er forskjellen rundt 1e-7.</p>
    <p>Det er derfor grensesnittet kan redesignes fritt: <b>så lenge motoren og eksporten står urørt,
    spiller flua nøyaktig de samme trekkene som før.</b></p>
  </section>

  <section class="slik-seksjon">
    <h2>Hva flua ikke er</h2>
    <ul class="punktliste">
      <li><b>Den er ikke en god sjakkspiller.</b> Den spiller fornuftige åpninger og posisjonssjakk,
          men gjør taktiske feil en klubbspiller straffer hardt. Med søk blir den en reell motstander,
          men den taper for enhver ordentlig sjakkmotor.</li>
      <li><b>Det er en forenklet hjerne, ikke en simulering.</b> Nevronene er enkle rate-enheter, det
          er én lekkasje per nevron, og det er ingen gap-koblinger, neuromodulering eller
          dendrittberegning. De 16 rundene er et fast antall, ikke fluas egen tidsfølelse.</li>
      <li><b>Den ser ikke brettet som en flue ser verden.</b> Modellen med øyne er et forsøk på å la
          sanseinngangen gå gjennom de ekte synsbanene, men en flue har selvsagt ingen oppfatning
          av sjakk i det hele tatt.</li>
      <li><b>Den er trent på mennesker, ikke på seg selv.</b> Modellen etterligner trekk fra rundt
          57 millioner stillinger spilt av mennesker på Lichess. Den har altså lært av menneskelig
          sjakk, formidlet gjennom et fluehjerne-formet nettverk.</li>
    </ul>
  </section>

  <section class="slik-seksjon">
    <h2>Hvor kommer delene fra?</h2>
    <p>Connectomet, celletypene, nevrotransmitterne og treningsdataene er andres arbeid, og skal
    krediteres som det. Selve modellen, oppskriften og treningen er beskrevet i
    <code>docs/SPEC.md</code>. Grensesnittet, den norske oversettelsen og denne versjonen av spillet
    er laget av <b>Oliver Lysø Hellevik</b>. Full oversikt står i <b>CREDITS.md</b>.</p>
  </section>

  <div class="slik-avslutning">
    <button class="btn btn-primary btn-big" id="btn-slik-spill" type="button">Spill mot flua</button>
  </div>
</div>`;

export const FEILSOK_INNHOLD = `
<h3>Fluesjakk finner ikke modellfilen</h3>
<p>Spillet trenger en modellfil på omtrent 120 MB. Den lastes ned én gang og lagres lokalt.
Åpne spillet på nytt og trykk <b>Last ned modellen</b>. Skjer det ingenting, sjekk at du er
tilkoblet internett.</p>

<h3>Flua svarer ikke</h3>
<p>Trykk <b>Prøv igjen</b> ved brettet. Det starter en frisk arbeidertråd og en ny vurdering av
stillingen. Partiet du har spilt, blir stående.</p>

<h3>Superflue bruker lang tid</h3>
<p>Det er meningen. Superflue gjør hundrevis av gjennomganger per trekk, og hver gjennomgang er et
helt framoverpass gjennom 134 209 nevroner. Har datamaskinen din et skjermkort som nettleseren får
bruke, går det merkbart raskere — det står hvilken motor som brukes i den tekniske informasjonen.
Velg <b>Flue</b> eller <b>Larve</b> hvis du vil ha kjappe svar.</p>

<h3>Spillet vil ikke starte i nettleseren</h3>
<p>Fluesjakk må kjøres fra en nettside eller fra den medfølgende appen, ikke ved å dobbeltklikke på
<code>index.html</code>. Nettlesere nekter å laste moduler og arbeidertråder fra lokale filer av
sikkerhetsgrunner. Bruk <b>Fluesjakk.exe</b> på Windows, eller kjør
<code>python -m http.server</code> i <code>web</code>-mappen og åpne adressen den skriver ut.</p>

<h3>Jeg vil se hva som faktisk skjer</h3>
<p>Slå på <b>Teknisk modus</b> i innstillingene. Da viser panelet ved siden av brettet modellnavn,
antall nevroner og koblinger, hvilken motor som kjører, verdien fra verdihodet, hvilket trekk som
ble valgt og hvor lang tid det tok.</p>

<h3>Jeg vil rapportere en feil</h3>
<p>Trykk <b>Kopier teknisk logg</b> nedenfor. Den inneholder nettleserversjon, modellinformasjon og
de siste hendelsene, uten noe personlig informasjon.</p>`;
