const pptxgen = require("pptxgenjs");

// Palette "Midnight Executive" -- academique, serieux, adapte a un contexte
// theorie/probabilites. Navy dominant, glace pour les fonds clairs, ambre
// comme accent net pour distinguer "adaptatif" de "aleatoire" dans les graphes.
const NAVY = "1E2761";
const NAVY_DARK = "141A45";
const ICE = "CADCFC";
const ICE_PALE = "F3F6FD";
const WHITE = "FFFFFF";
const AMBER = "E8A33D";
const GRAY = "9AA5B1";
const TEXT_DARK = "1B1F3B";
const TEXT_MUTED = "5B6472";

const HEADER = "Cambria";
const BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5

function slideHeader(slide, kicker, title) {
  slide.addText(kicker.toUpperCase(), {
    x: 0.6, y: 0.4, w: 8, h: 0.35,
    fontFace: BODY, fontSize: 12, bold: true, color: AMBER,
    charSpacing: 2, margin: 0,
  });
  slide.addText(title, {
    x: 0.6, y: 0.7, w: 11.5, h: 0.8,
    fontFace: HEADER, fontSize: 30, bold: true, color: TEXT_DARK,
    margin: 0,
  });
}

function iconCircle(slide, x, y, d, bg, letter, letterColor) {
  slide.addShape("ellipse", {
    x, y, w: d, h: d, fill: { color: bg }, line: { type: "none" },
  });
  slide.addText(letter, {
    x, y, w: d, h: d, align: "center", valign: "middle",
    fontFace: BODY, fontSize: d * 28, bold: true, color: letterColor,
    margin: 0,
  });
}

// ---------------------------------------------------------------------
// Slide 1 -- Titre
// ---------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  s.addShape("ellipse", {
    x: 10.6, y: -1.8, w: 5, h: 5,
    fill: { color: NAVY_DARK }, line: { type: "none" },
  });
  s.addShape("ellipse", {
    x: -1.5, y: 5.2, w: 4, h: 4,
    fill: { color: NAVY_DARK }, line: { type: "none" },
  });

  s.addText("PFA · UM6P VANGUARD CENTER / CMSIS", {
    x: 0.9, y: 2.0, w: 8, h: 0.4,
    fontFace: BODY, fontSize: 13, bold: true, color: AMBER, charSpacing: 2, margin: 0,
  });
  s.addText("Plateforme de quiz adaptatif", {
    x: 0.85, y: 2.5, w: 11, h: 1.3,
    fontFace: HEADER, fontSize: 44, bold: true, color: WHITE, margin: 0,
  });
  s.addText("Suivi Sprint 0 — moteur fonde sur la Knowledge Space Theory", {
    x: 0.9, y: 3.55, w: 10.5, h: 0.6,
    fontFace: BODY, fontSize: 20, color: ICE, margin: 0,
  });

  s.addShape("line", {
    x: 0.9, y: 4.35, w: 0, h: 1.5,
    line: { color: "3B4590", width: 1 },
  });
  s.addText([
    { text: "Encadrant : ", options: { bold: true, color: ICE } },
    { text: "Ahmed Ratnani\n", options: { color: WHITE } },
    { text: "Equipe : ", options: { bold: true, color: ICE } },
    { text: "S. Ibnjaa, S. Kharou, A. Zahir", options: { color: WHITE } },
  ], {
    x: 1.15, y: 4.35, w: 8, h: 0.9,
    fontFace: BODY, fontSize: 14, margin: 0, lineSpacingMultiple: 1.3,
  });
}

// ---------------------------------------------------------------------
// Slide 2 -- Le probleme / l'approche
// ---------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "Approche", "Une croyance qu'on affine, pas un score qu'on calcule");

  s.addText([
    { text: "Le systeme ne sait pas ce que l'etudiant sait.", options: { bold: true, breakLine: true, color: TEXT_DARK } },
    { text: " ", options: { breakLine: true } },
    { text: "Il maintient une ", options: {} },
    { text: "croyance", options: { bold: true, color: NAVY } },
    { text: " p, une distribution de probabilite sur les etats de connaissance possibles — pas un score unique.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "A chaque etape : poser la question qui reduit le plus l'incertitude, mettre a jour par Bayes, recommencer.", options: {} },
  ], {
    x: 0.6, y: 1.75, w: 5.5, h: 3.0,
    fontFace: BODY, fontSize: 15, color: TEXT_MUTED, margin: 0, lineSpacingMultiple: 1.35,
  });

  s.addShape("roundRect", {
    x: 0.6, y: 5.0, w: 5.5, h: 1.5, rectRadius: 0.08,
    fill: { color: ICE_PALE }, line: { type: "none" },
  });
  s.addText([
    { text: "p ∈ Δ(Z)", options: { bold: true, color: NAVY, fontSize: 22, breakLine: true } },
    { text: "le simplexe des distributions sur l'espace de connaissance Z", options: { color: TEXT_MUTED, fontSize: 12 } },
  ], {
    x: 0.9, y: 5.2, w: 5, h: 1.1, fontFace: BODY, margin: 0, lineSpacingMultiple: 1.2,
  });

  const layers = [
    ["1", "Combinatoire", "L'espace de connaissance Z, construit sur le graphe de prerequis"],
    ["2", "Probabilite", "Le modele de reponse BLIM et la mise a jour bayesienne de p"],
    ["3", "Geometrie", "Le simplexe n'est pas euclidien — metrique de Fisher, entropie"],
    ["4", "Controle", "Selection de la question par gain d'information (POMDP glouton)"],
  ];
  let ly = 1.75;
  for (const [num, title, desc] of layers) {
    iconCircle(s, 6.85, ly, 0.5, NAVY, num, WHITE);
    s.addText(title, {
      x: 7.55, y: ly - 0.05, w: 4.8, h: 0.35,
      fontFace: BODY, fontSize: 15, bold: true, color: TEXT_DARK, margin: 0,
    });
    s.addText(desc, {
      x: 7.55, y: ly + 0.28, w: 4.9, h: 0.65,
      fontFace: BODY, fontSize: 11.5, color: TEXT_MUTED, margin: 0, lineSpacingMultiple: 1.2,
    });
    ly += 1.15;
  }
}

// ---------------------------------------------------------------------
// Slide 3 -- Le defaut de design initial
// ---------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "Constat", "Le defaut de design initial");

  s.addText("Le moteur faisait : 1 concept = 1 question.", {
    x: 0.6, y: 1.7, w: 11.5, h: 0.5,
    fontFace: BODY, fontSize: 17, bold: true, color: TEXT_DARK, margin: 0,
  });

  // 8 concept boxes, chacun avec exactement une question
  const names = ["fractions", "equations1", "fonctions", "derivees", "limites", "integrales", "proba_base", "var_aleat."];
  const bx = 0.6, bw = 1.42, gap = 0.12, by = 2.5, bh = 0.9;
  names.forEach((n, i) => {
    const x = bx + i * (bw + gap);
    s.addShape("roundRect", {
      x, y: by, w: bw, h: bh, rectRadius: 0.06,
      fill: { color: ICE_PALE }, line: { color: ICE, width: 1 },
    });
    s.addText(n, {
      x, y: by + 0.08, w: bw, h: 0.35, align: "center",
      fontFace: BODY, fontSize: 9.5, bold: true, color: NAVY, margin: 0,
    });
    s.addShape("roundRect", {
      x: x + bw / 2 - 0.35, y: by + bh + 0.18, w: 0.7, h: 0.35, rectRadius: 0.5,
      fill: { color: GRAY }, line: { type: "none" },
    });
    s.addText("1 Q", {
      x: x + bw / 2 - 0.35, y: by + bh + 0.18, w: 0.7, h: 0.35, align: "center", valign: "middle",
      fontFace: BODY, fontSize: 10, bold: true, color: WHITE, margin: 0,
    });
  });

  s.addShape("roundRect", {
    x: 0.6, y: 4.35, w: 12.1, h: 2.0, rectRadius: 0.08,
    fill: { color: NAVY }, line: { type: "none" },
  });
  s.addText([
    { text: "Avec 8 questions disponibles pour 8 concepts, un quiz pose \"intelligemment\" et un quiz pose dans le desordre finissent au meme endroit : ", options: { color: ICE } },
    { text: "une fois toutes posees, il n'y a plus rien a optimiser.", options: { color: WHITE, bold: true } },
  ], {
    x: 1.0, y: 4.65, w: 7.3, h: 1.5,
    fontFace: BODY, fontSize: 15, margin: 0, lineSpacingMultiple: 1.35,
  });
  s.addText("8%", {
    x: 9.0, y: 4.55, w: 3.2, h: 1.0, align: "center",
    fontFace: HEADER, fontSize: 46, bold: true, color: AMBER, margin: 0,
  });
  s.addText("d'ecart adaptatif / aleatoire\nseulement — le defaut a corriger", {
    x: 9.0, y: 5.55, w: 3.2, h: 0.7, align: "center",
    fontFace: BODY, fontSize: 11, color: ICE, margin: 0, lineSpacingMultiple: 1.2,
  });
}

// ---------------------------------------------------------------------
// Slide 4 -- Le refactor concept/question
// ---------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "Refactor", "Separer le concept de la question");

  // Concept box (gauche)
  s.addShape("roundRect", {
    x: 0.8, y: 2.1, w: 3.4, h: 1.6, rectRadius: 0.08,
    fill: { color: NAVY }, line: { type: "none" },
  });
  s.addText([
    { text: "Concept", options: { bold: true, fontSize: 18, color: WHITE, breakLine: true } },
    { text: "noeud du graphe de prerequis\nZ se construit dessus", options: { fontSize: 11.5, color: ICE } },
  ], {
    x: 1.05, y: 2.3, w: 2.9, h: 1.2, fontFace: BODY, margin: 0, lineSpacingMultiple: 1.25,
  });

  // fleches + 3 questions (droite)
  const qy = [1.55, 2.55, 3.55];
  const originY = 2.9; // milieu vertical de la boite Concept
  qy.forEach((y, i) => {
    s.addShape("roundRect", {
      x: 6.3, y, w: 3.6, h: 0.8, rectRadius: 0.06,
      fill: { color: ICE_PALE }, line: { color: ICE, width: 1.25 },
    });
    s.addText(`Question ${i + 1}`, {
      x: 6.5, y: y + 0.08, w: 2.2, h: 0.3,
      fontFace: BODY, fontSize: 13, bold: true, color: TEXT_DARK, margin: 0,
    });
    s.addText("slip / guess propres", {
      x: 6.5, y: y + 0.42, w: 3.2, h: 0.3,
      fontFace: BODY, fontSize: 10.5, color: TEXT_MUTED, margin: 0,
    });
    const targetY = y + 0.4; // milieu vertical de la boite Question i
    const top = Math.min(originY, targetY);
    const h = Math.abs(targetY - originY);
    s.addShape("line", {
      x: 4.2, y: top, w: 2.1, h: h < 0.01 ? 0.01 : h,
      flipV: targetY < originY,
      line: { color: AMBER, width: 2 },
    });
  });

  s.addShape("roundRect", {
    x: 0.6, y: 5.35, w: 12.1, h: 1.35, rectRadius: 0.08,
    fill: { color: ICE_PALE }, line: { type: "none" },
  });
  s.addText([
    { text: "Plusieurs questions par concept, chacune avec son propre (slip, guess) ", options: { color: TEXT_DARK, bold: true } },
    { text: "→ la selection gloutonne a enfin un vrai choix a faire, meme apres avoir interroge chaque concept une fois.", options: { color: TEXT_MUTED } },
  ], {
    x: 1.0, y: 5.6, w: 11.3, h: 0.9,
    fontFace: BODY, fontSize: 14.5, margin: 0, lineSpacingMultiple: 1.3,
  });
}

// ---------------------------------------------------------------------
// Slide 5 -- Resultats chiffres (bar chart natif)
// ---------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "Resultats", "L'ecart devient significatif apres le refactor");

  const chartData = [
    {
      name: "Adaptatif (gain d'information)",
      labels: ["Avant\n(1 question / concept)", "Apres\n(3 questions / concept)"],
      values: [7.06, 11.53],
    },
    {
      name: "Aleatoire",
      labels: ["Avant\n(1 question / concept)", "Apres\n(3 questions / concept)"],
      values: [7.71, 18.18],
    },
  ];

  s.addChart("bar", chartData, {
    x: 0.6, y: 1.75, w: 7.6, h: 4.9,
    barDir: "col", barGapWidthPct: 40,
    chartColors: [AMBER, GRAY],
    showTitle: true, title: "Nombre moyen de questions posees",
    titleFontSize: 13, titleColor: TEXT_DARK, titleFontFace: BODY,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11,
    dataLabelColor: TEXT_DARK, dataLabelFormatCode: "0.0",
    catAxisLabelFontSize: 11, catAxisLabelColor: TEXT_MUTED, catAxisLabelFontFace: BODY,
    valAxisHidden: true,
    catGridLine: { style: "none" }, valGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: TEXT_MUTED,
    plotArea: { fill: { color: WHITE } }, chartArea: { fill: { color: WHITE } },
  });

  const stats = [
    ["37%", "de questions en moins\n(adaptatif vs aleatoire, apres refactor)"],
    ["65% → 76%", "de diagnostic exact\n(avant → apres)"],
    ["8% → 37%", "ecart adaptatif / aleatoire\n(avant → apres le refactor)"],
  ];
  let sy = 1.75;
  for (const [big, small] of stats) {
    s.addShape("roundRect", {
      x: 8.6, y: sy, w: 4.1, h: 1.35, rectRadius: 0.08,
      fill: { color: ICE_PALE }, line: { type: "none" },
    });
    s.addText(big, {
      x: 8.85, y: sy + 0.08, w: 3.6, h: 0.6,
      fontFace: HEADER, fontSize: 26, bold: true, color: NAVY, margin: 0,
    });
    s.addText(small, {
      x: 8.85, y: sy + 0.7, w: 3.6, h: 0.6,
      fontFace: BODY, fontSize: 10.5, color: TEXT_MUTED, margin: 0, lineSpacingMultiple: 1.15,
    });
    sy += 1.6;
  }
}

// ---------------------------------------------------------------------
// Slide 6 -- Mecanisme (line chart entropie)
// ---------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "Mecanisme", "La selection maximise la reduction d'incertitude");

  s.addText([
    { text: "IG(a;p) = H(p) − E", options: { italic: true } },
    { text: "y", options: { italic: true, subscript: true } },
    { text: "[H(p", options: { italic: true } },
    { text: "a", options: { italic: true, subscript: true } },
    { text: "y", options: { italic: true, superscript: true } },
    { text: ")]", options: { italic: true } },
    { text: "  —  a chaque question, on choisit celle qui fait chuter le plus l'entropie H(p) en esperance.", options: {} },
  ], {
    x: 0.6, y: 1.7, w: 12, h: 0.5,
    fontFace: BODY, fontSize: 14.5, color: TEXT_MUTED, margin: 0,
  });

  const adaptTrace = [4.087, 3.346, 3.189, 2.735, 1.807, 2.735, 2.711, 1.978, 1.416, 0.962, 1.422, 1.069, 0.91, 0.793];
  const randTrace = [4.087, 3.888, 3.649, 3.62, 3.609, 3.299, 3.127, 2.894, 2.358, 2.33, 2.758, 2.321, 2.307, 2.307, 2.291, 2.29, 2.287, 2.051, 1.729, 1.331, 1.163, 1.062, 0.662];
  const maxLen = randTrace.length;
  const cats = Array.from({ length: maxLen }, (_, i) => String(i));
  const padTo = (arr) => arr.concat(Array(maxLen - arr.length).fill(null));

  s.addChart("line", [
    { name: "Adaptatif (13 questions)", labels: cats, values: padTo(adaptTrace) },
    { name: "Aleatoire (22 questions)", labels: cats, values: randTrace },
  ], {
    x: 0.6, y: 2.35, w: 12.1, h: 4.3,
    chartColors: [AMBER, GRAY],
    lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 5,
    showTitle: false,
    catAxisTitle: "question posee (t)", showCatAxisTitle: true, catAxisTitleFontSize: 11,
    valAxisTitle: "entropie H(p_t)  [bits]", showValAxisTitle: true, valAxisTitleFontSize: 11,
    catAxisLabelFontSize: 9, catAxisLabelColor: TEXT_MUTED,
    valAxisLabelFontSize: 10, valAxisLabelColor: TEXT_MUTED,
    valGridLine: { color: "E8EAF0", size: 1 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendFontSize: 11, legendColor: TEXT_MUTED,
    plotArea: { fill: { color: WHITE } }, chartArea: { fill: { color: WHITE } },
  });
}

// ---------------------------------------------------------------------
// Slide 7 -- Rigueur : tests
// ---------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "Rigueur", "Une suite de tests, pas des print() dans __main__");

  const stats = [["44", "tests passent"], ["1", "xfail documente"], ["4", "couches couvertes"]];
  let sx = 0.6;
  for (const [big, small] of stats) {
    s.addShape("roundRect", {
      x: sx, y: 1.75, w: 2.55, h: 1.5, rectRadius: 0.08,
      fill: { color: NAVY }, line: { type: "none" },
    });
    s.addText(big, {
      x: sx, y: 1.9, w: 2.55, h: 0.7, align: "center",
      fontFace: HEADER, fontSize: 32, bold: true, color: AMBER, margin: 0,
    });
    s.addText(small, {
      x: sx, y: 2.65, w: 2.55, h: 0.5, align: "center",
      fontFace: BODY, fontSize: 12, color: ICE, margin: 0,
    });
    sx += 2.75;
  }

  s.addShape("roundRect", {
    x: 0.6, y: 3.65, w: 12.1, h: 2.9, rectRadius: 0.08,
    fill: { color: ICE_PALE }, line: { type: "none" },
  });
  s.addText("Regression corrigee : biais de smoothing dans le calcul Monte-Carlo", {
    x: 1.0, y: 3.95, w: 11.3, h: 0.4,
    fontFace: BODY, fontSize: 15, bold: true, color: TEXT_DARK, margin: 0,
  });
  s.addText([
    { text: "information_gain_mc", options: { fontFace: "Consolas", color: NAVY } },
    { text: " lissait uniquement le denominateur du KL (log(P / (Q+ε))) — un biais qui peut rendre l'estimateur negatif. Les deux distributions sont desormais lissees avant le calcul, puis renormalisees.", options: {} },
  ], {
    x: 1.0, y: 4.4, w: 11.3, h: 1.0,
    fontFace: BODY, fontSize: 13.5, color: TEXT_MUTED, margin: 0, lineSpacingMultiple: 1.3,
  });
  s.addText([
    { text: "Test de non-regression ajoute : ", options: { bold: true, color: TEXT_DARK } },
    { text: "avec un p ayant un zero exact au bord du simplexe (le pire cas pour ce biais), le gain d'information Monte-Carlo estime doit rester ≥ 0.", options: { color: TEXT_MUTED } },
  ], {
    x: 1.0, y: 5.5, w: 11.3, h: 0.9,
    fontFace: BODY, fontSize: 13.5, margin: 0, lineSpacingMultiple: 1.3,
  });
}

// ---------------------------------------------------------------------
// Slide 8 -- Feuille de route
// ---------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "Feuille de route", "Sprint 0 en cours, trois sprints devant");

  const sprints = [
    ["0", "Moteur pur", "Refactor concept/question, tests, benchmark. En cours — domaine reel a venir.", true],
    ["1", "API FastAPI", "POST /sessions, /answers, GET /diagnosis. Etat de session = vecteur p."],
    ["2", "Front Next.js", "Marginales par concept, barre de progression = 1 − H(p_t)/H(p0)."],
    ["3", "Calibration", "EM sur slip/guess a partir des logs, comparaison vs baseline IRT."],
  ];

  const sw = 2.95, gap = 0.13, sy = 1.9, sh = 4.2, sx0 = 0.6;
  sprints.forEach(([num, title, desc, active], i) => {
    const x = sx0 + i * (sw + gap);
    s.addShape("roundRect", {
      x, y: sy, w: sw, h: sh, rectRadius: 0.08,
      fill: { color: active ? NAVY : ICE_PALE }, line: { type: "none" },
    });
    s.addText(`Sprint ${num}`, {
      x: x + 0.25, y: sy + 0.3, w: sw - 0.5, h: 0.4,
      fontFace: BODY, fontSize: 13, bold: true,
      color: active ? AMBER : NAVY, margin: 0,
    });
    s.addText(title, {
      x: x + 0.25, y: sy + 0.75, w: sw - 0.5, h: 0.6,
      fontFace: HEADER, fontSize: 17, bold: true,
      color: active ? WHITE : TEXT_DARK, margin: 0,
    });
    s.addText(desc, {
      x: x + 0.25, y: sy + 1.4, w: sw - 0.5, h: 2.6,
      fontFace: BODY, fontSize: 11.5,
      color: active ? ICE : TEXT_MUTED, margin: 0, lineSpacingMultiple: 1.3,
    });
    if (i < sprints.length - 1) {
      s.addShape("ellipse", {
        x: x + sw + gap / 2 - 0.06, y: sy + sh / 2 - 0.06, w: 0.12, h: 0.12,
        fill: { color: AMBER }, line: { type: "none" },
      });
    }
  });
}

// ---------------------------------------------------------------------
// Slide 9 -- Cloture
// ---------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape("ellipse", {
    x: -2, y: -2.2, w: 5, h: 5,
    fill: { color: NAVY_DARK }, line: { type: "none" },
  });
  s.addText("Merci", {
    x: 0.9, y: 2.7, w: 8, h: 1.1,
    fontFace: HEADER, fontSize: 44, bold: true, color: WHITE, margin: 0,
  });
  s.addText("Questions et retours bienvenus", {
    x: 0.95, y: 3.7, w: 8, h: 0.6,
    fontFace: BODY, fontSize: 18, color: ICE, margin: 0,
  });
  s.addText("Plateforme de quiz adaptatif — PFA UM6P Vanguard Center / CMSIS", {
    x: 0.95, y: 6.8, w: 10, h: 0.4,
    fontFace: BODY, fontSize: 11, color: "6B75AE", margin: 0,
  });
}

pres.writeFile({ fileName: "sprint0_suivi.pptx" }).then(() => {
  console.log("ecrit : sprint0_suivi.pptx");
});
