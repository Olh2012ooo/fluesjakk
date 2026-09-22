// i18n.js — all synlig tekst i Fluesjakk, på norsk bokmål.
//
// Alt brukergrensesnittet henter tekstene sine herfra. Det gjør det enkelt å se over språket
// samlet, og det er forutsetningen for at en fremtidig engelsk versjon bare blir en ny ordbok.
//
// Skrivestil: naturlig norsk, ikke oversatt ord for ord. Sjakkuttrykk følger norsk sjakktradisjon
// (sjakk, sjakk matt, patt, remis, bondeforvandling, springer, løper, tårn).

export const SPRAAK = 'nb';

// ---------------------------------------------------------------- generelt
export const T = {
  // merkevare
  navn: 'Fluesjakk',
  slagord: 'Sjakk mot en flue',

  // navigasjon
  start: 'Start spill',
  nyttParti: 'Nytt parti',
  innstillinger: 'Innstillinger',
  omSpillet: 'Om spillet',
  slikFungererDet: 'Slik fungerer det',
  tilbake: 'Tilbake',
  lukk: 'Lukk',
  avbryt: 'Avbryt',
  lagre: 'Lagre',
  spillerIgjen: 'Spill igjen',
  giOpp: 'Gi opp',
  angre: 'Angre',
  snuBrettet: 'Snu brettet',
  kopierPgn: 'Kopier PGN',
  kopier: 'Kopier',
  kopiert: 'Kopiert',
  provIgjen: 'Prøv igjen',
  feilsoking: 'Feilsøking',

  // farger
  hvit: 'Hvit',
  svart: 'Svart',
  tilfeldig: 'Tilfeldig',
  velgSide: 'Velg side',
  duSpiller: 'Du spiller',

  // vanskelighetsgrader
  vanskelighetsgrad: 'Vanskelighetsgrad',
  larve: 'Larve',
  flue: 'Flue',
  superflue: 'Superflue',

  // brett og status
  dinTur: 'Din tur',
  fluasTur: 'Fluas tur',
  fluaTenker: 'Flua tenker …',
  sjakk: 'Sjakk',
  sjakkMatt: 'Sjakk matt',
  patt: 'Patt',
  remis: 'Remis',
  duVant: 'Du vant',
  duTapte: 'Du tapte',
  velgBrikke: 'Velg brikke',
  bondeforvandling: 'Bondeforvandling',
  sisteTrekk: 'Siste trekk',
  lovligeTrekk: 'Lovlige trekk',

  // panel
  trekkliste: 'Trekkliste',
  flueaktivitet: 'Flueaktivitet',
  fluasHumor: 'Fluas humør',
  fluasOyne: 'Fluas øyne',
  seer: 'Ser',
  foler: 'Føler',
  tenketid: 'Tenketid',
  sisteTenketid: 'Siste',
  totalTenketid: 'Totalt',
  sok: 'Søk',
  simuleringer: 'simuleringer',
  ingenTrekk: 'Ingen trekk ennå',

  // humør
  humor: {
    tenker: 'tenker',
    trygg: 'trygg',
    selvgod: 'selvgod',
    nervos: 'nervøs',
    panikk: 'panikk',
    nysgjerrig: 'nysgjerrig',
  },

  // parti
  dittParti: 'Ditt parti',
  mot: 'mot',
  du: 'Du',
  flua: 'Flua',
  velgMotstander: 'Hvordan skal flua spille?',
  fortsett: 'Fortsett',
  dittNavn: 'Navnet ditt',
  navnHjelp: 'Brukes på topplisten',
  spiller: 'Spiller',

  // resultat
  partietErFerdig: 'Partiet er ferdig',
  resultat: {
    vinner: 'Du slo flua',
    tap: 'Flua slo deg',
    remis: 'Partiet endte remis',
    oppgitt: 'Du ga opp',
  },
  antallTrekk: 'trekk',
  varighet: 'Varighet',

  // deling
  delResultat: 'Del resultatet',
  kopierBilde: 'Kopier bilde',
  lastNedBilde: 'Last ned bilde',
  kopierTekst: 'Kopier tekst',

  // toppliste
  toppliste: 'Toppliste',
  topplisteTom: 'Ingen partier ennå. Slå flua og kom tilbake.',
  spillernavn: 'Spiller',
  resultatKol: 'Resultat',
  vanskelighetskol: 'Mot',
  trekkKol: 'Trekk',
  tidKol: 'Tid',
  datoKol: 'Dato',
  tomListe: 'Tøm listen',

  // vennelag / pass-og-spill
  vennelag: 'Vennelag',
  vennelagHjelp: 'Alle spiller mot samme flue. Den som slår flua på færrest trekk vinner kvelden.',
  spillereEnPerLinje: 'Spillere, én per linje',
  startTurneringen: 'Start turneringen',
  spillerNa: 'spiller nå',
  venter: 'venter',
  sloFlua: 'slo flua',
  tapte: 'tapte',
  uavgjort: 'uavgjort',
  avsluttVennelag: 'Avslutt vennelaget',
  ingenSloFlua: 'Ingen slo flua. Flua vinner kvelden.',
  sloFluaRaskest: (navn) => `${navn} slo flua raskest.`,
  antallSpilt: (a, b) => `${a} av ${b} har spilt`,

  // last
  vokserHjerne: 'Vokser fluehjernen …',
  henterModell: 'Henter modellen …',
  pakkerUt: 'Pakker ut …',
  koblerTil: 'Kobler til …',
  lagrer: 'Lagrer …',
  fraHurtiglager: 'fra hurtiglager',
  klarTilAAspille: 'Spill mot flua',
  ingenModell: 'Fant ingen modell',
  lasterNed: 'Laster ned modellen',

  // innstillinger
  utseende: 'Utseende',
  spill: 'Spill',
  lys: 'Lys',
  mork: 'Mørk',
  system: 'Følg systemet',
  bretttema: 'Bretttema',
  temaKveld: 'Kveld',
  temaSkifer: 'Skifer',
  temaSkog: 'Skog',
  animasjoner: 'Animasjoner',
  redusertBevegelse: 'Redusert bevegelse',
  visLovligeTrekk: 'Vis lovlige trekk',
  visSisteTrekk: 'Vis siste trekk',
  visKoordinater: 'Vis koordinater',
  lyd: 'Lyd',
  fluasKommentarer: 'Fluas kommentarer',
  visFlueaktivitet: 'Vis flueaktivitet',
  visFluasOyne: 'Vis fluas øyne',
  tekniskModus: 'Teknisk modus',
  visTeknisk: 'Vis teknisk informasjon',
  nullstillInnstillinger: 'Nullstill innstillinger',
  innstillingerLagret: 'Innstillingene er lagret',

  // teknisk modus
  teknisk: 'Teknisk',
  modell: 'Modell',
  nevroner: 'Nevroner',
  forbindelser: 'Forbindelser',
  synapser: 'Synapser',
  tidssteg: 'Tidssteg',
  motor: 'Motor',
  fotoreseptorer: 'Fotoreseptorer',
  fremoverTid: 'Framover',
  perTidssteg: 'Per tidssteg',
  verdi: 'Verdi',
  valgtTrekk: 'Valgt trekk',
  beregningstid: 'Beregningstid',
  treffsikkerhet: 'Policy-topp',
  visNevronaktivitet: 'Vis nevronaktivitet',
  flaskehals: 'Kjøremotor',

  // ---- feilmeldinger (menneskelige, aldri stakksporinger) ----
  feil: {
    ingenModellTittel: 'Fluesjakk finner ikke modellfilen',
    ingenModell: 'Spillet trenger én modellfil før du kan begynne. Du kan laste den ned nå — det gjøres bare én gang.',
    modellKorrupt: 'Modellfilen ser ut til å være ødelagt. Prøv å laste den ned på nytt.',
    nettverk: 'Fikk ikke kontakt med modellserveren. Sjekk internettforbindelsen og prøv igjen.',
    nettverkTittel: 'Fikk ikke lastet ned modellen',
    avbrutt: 'Nedlastingen ble avbrutt.',
    arbeider: 'Fluehjernen sluttet å svare. Prøv å starte spillet på nytt.',
    tidsavbrudd: 'Flua brukte for lang tid på å svare.',
    ingenPlass: 'Det er ikke plass til modellfilen på denne datamaskinen.',
    ukjent: 'Noe gikk galt.',
    hjelp: 'Hvis problemet fortsetter, se Feilsøking i Om spillet.',
    apneLogg: 'Åpne teknisk logg',
  },

  // ---- nedlasting ----
  nedlasting: {
    tittel: 'Velkommen til Fluesjakk',
    forklaring: 'Spillet trenger én modellfil før du kan begynne. Filen inneholder fluehjernen — 134 209 nevroner og 2,7 millioner koblinger.',
    knapp: 'Last ned modellen',
    igjen: 'Det gjøres bare én gang.',
    fremdrift: (pct) => `${pct} %`,
    storrelse: (mb) => `${mb} MB`,
    ferdig: 'Modellen er klar.',
    spillKnapp: 'Spill',
    lisens: 'Modellen er avledet av FlyWire-connectomet og er lisensiert under CC BY-NC 4.0. Den kan brukes fritt til ikke-kommersielle formål.',
    lisensLenke: 'Les mer om lisensen',
    avbryt: 'Avbryt nedlastingen',
    fortsetter: 'Fortsetter …',
    starter: 'Starter …',
  },

  // ---- om ----
  om: {
    tittel: 'Om Fluesjakk',
    utvikler: 'Utvikler',
    utviklerNavn: 'Oliver Lysø Hellevik',
    byggetPa: 'Bygget på',
    takk: 'Takk til',
    versjon: 'Versjon',
    personvern: 'Personvern',
    personvernTekst: 'Fluesjakk kjører helt og holdent på din egen datamaskin. Spillet har ingen konto, sender ingen data til noen server og inneholder ingen sporing. Modellen lastes ned én gang og lagres lokalt; etter det virker spillet uten internett.',
    systemkrav: 'Systemkrav',
    tastatursnarveier: 'Tastatursnarveier',
    hurtigtaster: {
      snu: 'Snu brettet',
      angre: 'Angre siste trekk',
      nytt: 'Nytt parti',
      lukk: 'Lukk dialog',
      piler: 'Flytt markøren på brettet',
      enter: 'Velg brikke eller slipp den',
      escape: 'Avbryt valg',
    },
  },
};

// ---------------------------------------------------------------- vanskelighetsgrader
/**
 * De tre nivåene, i den rekkefølgen de vises.
 *
 * `id` er navnet grensesnittet bruker (norsk). `motor` er nøkkelen arbeideren kjenner igjen
 * (SPEC §9) — den er engelsk fordi den er en del av protokollen mellom siden og motoren, ikke av
 * det brukeren ser. De to navnerommene holdes bevisst atskilt her, så en norsk omskriving av
 * grensesnittet aldri kan endre hvilket nivå som faktisk spilles.
 */
export const VANSKELIGHETER = [
  {
    id: 'larve',
    motor: 'larva',
    navn: T.larve,
    ikon: '🐛',
    kort: 'Uforutsigbar og svak',
    beskrivelse: 'Flua følger bare magefølelsen. Den trekker et tilfeldig trekk fra policyen uten å se framover — spillbart, kaotisk og lett å slå.',
  },
  {
    id: 'flue',
    motor: 'fly',
    navn: T.flue,
    ikon: '🪰',
    kort: 'Fluas normale spill',
    beskrivelse: 'Fluas egen favoritt. Den prøver de tre beste kandidatene ett steg fram og lar verdihodet avgjøre hvilket trekk motstanderen liker dårligst.',
  },
  {
    id: 'superflue',
    motor: 'superfly',
    navn: T.superflue,
    ikon: '🧠',
    kort: 'Flua får tenke lenger',
    beskrivelse: 'Samme hjerne, men nå får den tid. Den søker gjennom hundrevis av varianter med Monte-Carlo-tresøk, der hver eneste stilling vurderes av fluehjernen.',
  },
];

/** Norsk nivånavn → nøkkelen arbeideren forstår. */
export const VANSKELIGHET_MOTOR = Object.fromEntries(VANSKELIGHETER.map((v) => [v.id, v.motor]));

// ---------------------------------------------------------------- humør
/** Verdihodets utsagn mappes til et humør. Grensene er de samme som i motoren. */
export function humorFor(verdi) {
  if (verdi > 0.55) return 'selvgod';
  if (verdi > 0.18) return 'trygg';
  if (verdi < -0.55) return 'panikk';
  if (verdi < -0.18) return 'nervos';
  return 'nysgjerrig';
}

// ---------------------------------------------------------------- kommentarer
const velg = (liste, rnd = Math.random) => liste[Math.floor(rnd() * liste.length) % liste.length];
const prosent = (p) => Math.round(p * 100);

/**
 * Fluas kommentar til et trekk. Alle linjene bygger på det nettverket faktisk rapporterte
 * (policy-topp, verdihode, antall simuleringer) — ingenting er diktet opp fra sjakkheuristikk.
 * @param {{policyTop?:{san:string,p:number}[], value:number, san?:string, vanskelighet:string, sims?:number}} o
 * @param {() => number} rnd  injiserbar tilfeldighet, for testbarhet
 */
export function kommentar(o, rnd = Math.random) {
  const { policyTop, value, san, vanskelighet, sims } = o;
  const topp = policyTop?.[0];
  const nest = policyTop?.[1];
  const pct = topp ? prosent(topp.p) : 0;
  const linjer = [];

  if (topp && san && topp.san === san) {
    if (vanskelighet === 'superflue' && sims) {
      linjer.push(velg([
        `Etter ${sims} simuleringer holder flua fortsatt på ${san} (${pct} % på magefølelsen).`,
        `Instinktet og søket er enige: ${san}.`,
        `Søket bekreftet magefølelsen. ${san}.`,
      ], rnd));
    } else if (pct >= 70) {
      linjer.push(velg([
        `Flua er ${pct} % sikker på ${san}.`,
        `${san}, selvfølgelig. ${pct} % av flua er enig med seg selv.`,
        `${pct} % av de synkende nevronene ville spille ${san}.`,
      ], rnd));
    } else if (pct >= 40) {
      linjer.push(velg([
        `${san} — fluas favoritt, med ${pct} %.`,
        `Flua heller mot ${san} (${pct} %).`,
      ], rnd));
    } else {
      linjer.push(velg([
        `Flua vinglet mellom ${topp.san} og ${nest?.san || '…'}, og landet på ${san}.`,
        `Bare ${pct} % sikker, men flua spiller ${san} likevel.`,
      ], rnd));
    }
  } else if (san && topp) {
    if (vanskelighet === 'superflue' && sims) {
      linjer.push(velg([
        `Instinktet sa ${topp.san}, men etter ${sims} simuleringer foretrekker flua ${san}.`,
        `Søket overtalte flua bort fra ${topp.san}. Det blir ${san}.`,
      ], rnd));
    } else if (vanskelighet === 'larve') {
      linjer.push(velg([
        `Larven kastet terningen og fikk ${san}.`,
        `${san}! Flua var ikke sikker selv — ${pct} % ville ha ${topp.san}.`,
      ], rnd));
    } else {
      linjer.push(velg([
        `Flua likte ${topp.san}, men verdihodet satte foten ned. ${san} i stedet.`,
        `Ett trekk med tvil gjorde ${topp.san} om til ${san}.`,
      ], rnd));
    }
  } else if (topp) {
    linjer.push(pct >= 60
      ? `Flua liker allerede ${topp.san} (${pct} %).`
      : `Flua er i tvil mellom ${topp.san} og ${nest?.san || '…'}.`);
  }

  if (value > 0.55) linjer.push(velg(['Den lukter seier.', 'Gnikker forbeina sammen.', 'Den tror den vinner. Det kan hende den har rett.'], rnd));
  else if (value > 0.18) linjer.push(velg(['Den liker dette.', 'Vingene er avslappet.', 'Stille selvsikker.'], rnd));
  else if (value < -0.55) linjer.push(velg(['Flua aner fare.', 'Panikk i sopplegemene.', 'Hvert eneste synkende nevron skriker.'], rnd));
  else if (value < -0.18) linjer.push(velg(['Flua er nervøs.', 'Antennene dirrer.', 'Den liker ikke denne stillingen.'], rnd));

  return linjer.join(' ');
}

/** Teksten som vises rett etter et trekk, fra humøret alene. */
export function humorTekst(humor, rnd = Math.random) {
  const M = {
    selvgod: ['Flua har bestemt seg for at den vinner.', 'Ingen tvil å spore.', 'Den ser allerede for seg neste trekk.'],
    trygg: ['Rolig og fornøyd.', 'Den liker stillingen sin.', 'Vingene hviler.'],
    nysgjerrig: ['Flua følger med på brettet.', 'Den venter på ditt trekk.', 'Antennene vender mot brettet.'],
    nervos: ['Den liker ikke dette.', 'Antennene dirrer svakt.', 'Flua leter etter en vei ut.'],
    panikk: ['Full panikk i fluehjernen.', 'Den vurderer å flykte.', 'Alle synkende nevroner skriker samtidig.'],
    tenker: ['Flua tenker …', 'Nevronene jobber.', 'Flua regner på stillingen.'],
  };
  return velg(M[humor] || M.nysgjerrig, rnd);
}

// ---------------------------------------------------------------- resultattekster
export function resultatTekst(utfall, navn, grunn) {
  const arsak = {
    checkmate: 'sjakk matt',
    stalemate: 'patt',
    repetition: 'trekk som gjentas tre ganger',
    'threefold repetition': 'trekk som gjentas tre ganger',
    material: 'for lite materiell til å vinne',
    'insufficient material': 'for lite materiell til å vinne',
    fifty: 'femti-trekksregelen',
    'fifty-move rule': 'femti-trekksregelen',
    resignation: 'du ga opp',
  }[grunn] || grunn;

  if (utfall === 'vinner') return { tittel: `${navn} slo flua`, under: `Seier på ${arsak}.` };
  if (utfall === 'tap') return { tittel: 'Flua vant', under: `Tap på ${arsak}.` };
  return { tittel: 'Remis', under: `Uavgjort på ${arsak}.` };
}

/** Delingstekst til utklippstavlen. */
export function delTekst({ utfall, vanskelighet, antallTrekk, varighet, adresse }) {
  const hvem = utfall === 'vinner'
    ? 'Jeg slo en fluehjerne i sjakk'
    : utfall === 'tap'
      ? 'En fluehjerne slo meg i sjakk'
      : 'Jeg spilte remis mot en fluehjerne';
  return `${hvem} — ${vanskelighet}, ${antallTrekk} trekk, ${varighet}. ${adresse}`;
}

// ---------------------------------------------------------------- formatering
export const tall = (n) => Number(n).toLocaleString('nb-NO');
export const ms = (v) => (v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(1)} s`);
export const klokke = (v) => {
  const s = Math.round(v / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};
export const mb = (bytes) => (bytes / 1e6).toFixed(1);

// ---------------------------------------------------------------- brikkenavn
export const BRIKKENAVN = {
  p: 'bonde', n: 'springer', b: 'løper', r: 'tårn', q: 'dronning', k: 'konge',
};
