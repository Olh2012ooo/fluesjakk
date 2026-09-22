// Fluesjakk — skrivebordsapp.
//
// Appen er den samme nettsiden som ligger i web/, pakket i et eget vindu. Hvorfor en liten lokal
// tjener i stedet for å åpne index.html direkte: nettlesere (og Electron) nekter å laste
// ES-moduler, Web Workers og Cache API fra file://. En tjener på 127.0.0.1 løser det uten at noe
// går ut på nettet — den lytter bare på lokalmaskinen og serverer filer fra denne pakken.
//
// Ingenting lastes ned her. Modellen hentes av siden selv, første gang, og legges i nettleserens
// hurtiglager i brukerprofilen. Er modellen lagt ved siden av spillet (web/model/), brukes den og
// ingenting hentes.

const { app, BrowserWindow, Menu, dialog, shell, nativeTheme } = require('electron');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const ER_PAKKET = app.isPackaged;
/** web/ ligger ved siden av kildekoden i utvikling, og under resources/ i en pakket app. */
const WEB_DIR = ER_PAKKET ? path.join(process.resourcesPath, 'web') : path.join(__dirname, '..', 'web');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.woff2': 'font/woff2',
  '.flyb': 'application/octet-stream',
  '.gz': 'application/gzip',
};

let vindu = null;
let tjener = null;

/**
 * Serverer web/ på en ledig port på 127.0.0.1. Stien løses opp og kontrolleres mot rotmappen, så
 * en forespørsel aldri kan lese filer utenfor spillet.
 */
function startTjener() {
  return new Promise((resolve, reject) => {
    tjener = http.createServer((req, res) => {
      let sti;
      try {
        sti = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname);
      } catch {
        res.writeHead(400).end('ugyldig adresse');
        return;
      }
      const fil = path.join(WEB_DIR, sti === '/' ? 'index.html' : sti);
      const rot = path.resolve(WEB_DIR);
      if (!path.resolve(fil).startsWith(rot + path.sep) && path.resolve(fil) !== rot) {
        res.writeHead(403).end('utenfor spillet');
        return;
      }
      fs.stat(fil, (feil, stat) => {
        if (feil || !stat.isFile()) { res.writeHead(404).end('ikke funnet'); return; }
        res.writeHead(200, {
          'content-type': MIME[path.extname(fil).toLowerCase()] || 'application/octet-stream',
          'content-length': stat.size,
          'cache-control': 'no-cache',
        });
        fs.createReadStream(fil).pipe(res);
      });
    });
    tjener.on('error', reject);
    tjener.listen(0, '127.0.0.1', () => resolve(tjener.address().port));
  });
}

function meny() {
  const mal = [
    {
      label: 'Fil',
      submenu: [
        { label: 'Nytt parti', accelerator: 'CmdOrCtrl+N', click: () => vindu?.webContents.executeJavaScript('window.fluesjakk && window.fluesjakk.visStart()') },
        { type: 'separator' },
        { label: 'Skriv ut …', accelerator: 'CmdOrCtrl+P', click: () => vindu?.webContents.print() },
        { type: 'separator' },
        { label: 'Avslutt', role: 'quit' },
      ],
    },
    {
      label: 'Vis',
      submenu: [
        { label: 'Start på nytt', accelerator: 'CmdOrCtrl+R', role: 'reload' },
        { label: 'Faktisk størrelse', role: 'resetZoom' },
        { label: 'Zoom inn', role: 'zoomIn' },
        { label: 'Zoom ut', role: 'zoomOut' },
        { type: 'separator' },
        { label: 'Fullskjerm', role: 'togglefullscreen' },
        { label: 'Verktøy for utviklere', accelerator: 'CmdOrCtrl+Shift+I', role: 'toggleDevTools' },
      ],
    },
    {
      label: 'Hjelp',
      submenu: [
        {
          label: 'Åpne mappen for modellfiler',
          click: () => shell.openPath(app.getPath('userData')),
        },
        {
          label: 'Om Fluesjakk',
          click: () => {
            dialog.showMessageBox(vindu, {
              type: 'info',
              title: 'Om Fluesjakk',
              message: `Fluesjakk ${app.getVersion()}`,
              detail: 'Sjakk mot et nevralt nettverk koblet som hjernen til en fruktflue.\n\n'
                + 'Laget og videreutviklet av Oliver Lysø Hellevik.\n\n'
                + 'Connectomet er FlyWire (Dorkenwald et al. 2024, Schlegel et al. 2024), CC BY-NC 4.0.\n'
                + 'Koden er MIT-lisensiert. Kjører helt lokalt — ingen konto, ingen sporing.',
              buttons: ['Lukk'],
            });
          },
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(mal));
}

async function opprettVindu() {
  const port = await startTjener();

  vindu = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 380,
    minHeight: 520,
    show: false,
    backgroundColor: '#08090a',
    title: 'Fluesjakk',
    autoHideMenuBar: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      // WebGPU brukes til framoverpasset når skjermkortet og driveren tillater det
      enableBlinkFeatures: 'WebGPU',
    },
  });

  vindu.once('ready-to-show', () => vindu.show());
  vindu.on('page-title-updated', (e) => e.preventDefault());   // tittelen skal stå fast

  // eksterne lenker åpnes i systemets nettleser, ikke inne i spillet
  vindu.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });

  await vindu.loadURL(`http://127.0.0.1:${port}/index.html`);
  if (process.env.FLUESJAKK_SMOKE) await royktest();
  return port;
}

/**
 * Røyktest for CI og pakkebygging: venter på at spillet har lastet (eller tilbudt nedlasting),
 * melder fra på stdout og avslutter med 0 bare hvis siden faktisk kom opp.
 * Slås på med FLUESJAKK_SMOKE=1, og påvirker ingenting ellers.
 */
async function royktest() {
  const svar = await vindu.webContents.executeJavaScript(`(async () => {
    const vent = (f, ms) => new Promise((ok) => { const t0 = Date.now();
      (function sjekk() { if (f()) return ok(true); if (Date.now() - t0 > ms) return ok(false); setTimeout(sjekk, 200); })(); });
    const klar = await vent(() => !document.getElementById('btn-start')?.disabled
      || !document.getElementById('panel-nedlasting')?.hidden, 60000);
    const h = window.fluesjakk?.hjerne?.info?.header || {};
    return { klar, tittel: document.title, startknapp: document.getElementById('btn-start')?.textContent,
      nevroner: h.n || null, modell: h.run_name || null, vanskelighetskort: document.querySelectorAll('#vanskelighetskort .kort').length };
  })()`);
  console.log('FLUESJAKK_SMOKE ' + JSON.stringify(svar));
  app.exit(svar.klar && svar.vanskelighetskort === 3 ? 0 : 1);
}

// Bare ett spill om gangen: en ny oppstart skal fokusere det som kjører.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!vindu) return;
    if (vindu.isMinimized()) vindu.restore();
    vindu.focus();
  });

  app.whenReady().then(async () => {
    nativeTheme.themeSource = 'dark';
    meny();
    try {
      await opprettVindu();
    } catch (feil) {
      dialog.showErrorBox('Fluesjakk kunne ikke starte',
        `Fant ikke spillfilene i:\n${WEB_DIR}\n\n${feil && feil.message ? feil.message : feil}`);
      app.quit();
    }

    app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) opprettVindu(); });
  });

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
  });

  app.on('will-quit', () => { try { tjener?.close(); } catch { /* ignorer */ } });
}
