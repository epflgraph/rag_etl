import logging
from pathlib import Path

import re

from typing import List
from pydantic import BaseModel, Field

from rag_etl.utils.llms import send_llm_request

from rag_etl.config import CONFIG


def split_by_most_common_heading(md_text: str):
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)', re.MULTILINE)
    matches = heading_pattern.findall(md_text)

    if not matches:
        logging.warning(f"No headings found in Markdown code. Trying without splitting...")
        return [md_text]

    # Count heading levels
    counts = {}
    for hashes, _ in matches:
        level = len(hashes)
        counts[level] = counts.get(level, 0) + 1

    # Find most common level
    most_common_level = max(counts, key=counts.get)

    # Split on that level (keep headings using lookahead)
    split_pattern = re.compile(
        rf'(?=^#{{{most_common_level}}}\s+)',
        re.MULTILINE
    )

    sections = split_pattern.split(md_text)

    return [s.strip() for s in sections if s.strip()]


def merge_up_to_max_lines(sections, max_lines):
    if len(sections) < 2:
        return sections

    first = sections[0]
    second = sections[1]

    if len(first.splitlines()) + len(second.splitlines()) <= max_lines:
        new_first = '\n'.join([first, second])
        new_sections = [new_first] + sections[2:]
        return merge_up_to_max_lines(new_sections, max_lines)

    return [first] + merge_up_to_max_lines(sections[1:], max_lines)


def split_md_into_exercises(md_path, exercises_path):
    # Normalise to Paths
    md_path = Path(md_path)
    exercises_path = Path(exercises_path)

    # Prepare system prompt
    system_prompt = """
You are a careful Markdown document segmenter.
Your task is to read a long Markdown file, annotated with line numbers, containing multiple exercises and split it into separate chunks, one per exercise.
Do that by finding the correct lines that delimit every exercise. 
You also need to identify the exercise numbers as well as whether the chunks correspond to the exercise statement or the solution.  

Rules:
- Infer exercise boundaries according to headings. Typically, exercises start with a heading containing the text "Exercise N", "Problem N" or even "Solution N".
- The lines you produce to delimit an exercise should be included in the exercise (as in closed intervals). For example, if "Exercise 1.1" and "Exercise 1.2" are consecutive and there is nothing in between, then "Exercise 1.1" should run until line N and "Exercise 1.2" should start at line N+1.     
- Ignore any introductory or preface material that is not related to any exercise.  
- If an exercise (e.g. "Exercise 1") appears more than once (for instance, this can happen with the exercise statement and solution), produce a chunk for each occurrence.
- Think of the tuple (`exercise_number`, `is_solution`) as a sort of id for exercises.
- You may produce several chunks with the same number and is_solution, for example when an exercise is cut across two pages and every page has a header (in this case produce two exercise blocks that skip the header in between).

Example:
For this Markdown document:
```
[L1] # École Polytechnique Fédérale de Lausanne EPFL  
[L2] ## Section de Génie Civil  
[L3] ### Laboratoire du Génie Parasismique et Dynamique des Structures EESD  
[L4] #### Statique I (CIVIL-124)
[L5] 
[L6] ## Corrigé Semaine 2
[L7] 
[L8] ### Chapitre 3 : 3.10.10, 3.10.12, 3.10.17, 3.10.18, 3.10.20
[L9] 
[L10] #### Exercice additionnel 2 : Exercice barrage de la Grande Dixence
[L11] 
[L12] ![Diagram showing triangular load distribution and calculations for reactions](#)
[L13] 
[L14] Barrage poids :
[L15] 
[L16] ![Diagram illustrating forces acting on a gravity dam including hydrostatic pressure and weight](#)
[L17] 
[L18] La pression hydrostatique à une profondeur $x$ est:
[L19] 
[L20] $$
[L21] p = \\rho \\cdot g \\cdot x
[L22] $$
[L23] 
[L24] La pression hydrostatique impose donc une charge répartie avec la profondeur. La pression maximale (à la base):
[L25] 
[L26] $$
[L27] p_{\\text{max}} = \\rho \\cdot g \\cdot h
[L28] $$
[L29] 
[L30] La pression dans le sol: on assume, pour simplifier, une distribution linéaire des pressions dans le sol. La valeur au contact de l'eau doit être $p_{\\text{max}}$, la valeur au point O doit être 0 (sinon on a des pontons d'eau).
[L31] 
[L32] ---
[L33] 
[L34] On suppose que l'on ne peut pas avoir de glissements entre la structure et le sol. Le seul mécanisme possible est le renversement autour du point O.
[L35] 
[L36] ![Mechanism of a wall viewed from above](#)  
[L37] ![Cross-section of a wall with forces applied](#)  
[L38] ![Forces acting on a wall](#)
[L39] 
[L40] Réduction des forces distribuées aux forces concentrées:  
[L41] $$
[L42] H = \\frac{A}{2} \\cdot \\rho_{\\text{max}} \\cdot h = \\frac{A}{2} \\cdot \\rho_{\\text{eg}} \\cdot h^2
[L43] $$
[L44] Point d'application : $x_H = \\frac{h}{3}$
[L45] 
[L46] $$
[L47] V = \\frac{A}{2} \\cdot \\rho_{\\text{max}} \\cdot b = \\frac{A}{2} \\cdot \\rho_{\\text{eg}} \\cdot h \\cdot b
[L48] $$
[L49] Point d'application : $x_V = \\frac{2}{3} \\cdot b$
[L50] 
[L51] $$
[L52] G = \\frac{A}{2} \\cdot (\\rho_{\\text{eg}} \\cdot g) \\cdot h \\cdot b = \\frac{A}{2} \\cdot (2.5 \\cdot \\rho_{\\text{eg}}) \\cdot h \\cdot b = \\frac{5}{4} \\cdot \\rho_{\\text{eg}} \\cdot h \\cdot b
[L53] $$
[L54] $x_G = \\frac{2}{3} \\cdot b$
[L55] 
[L56] Condition de renversement : $\\Sigma M_{O,t} \\leq 0$ (limite : $\\Sigma M_{O,t} = 0$)
[L57] 
[L58] $$
[L59] \\Sigma M_{O,t} = 0 \\Rightarrow - H \\cdot \\frac{h}{3} - V \\cdot \\frac{2}{3} \\cdot b + G \\cdot \\frac{2}{3} \\cdot b = 0
[L60] $$
[L61] 
[L62] $$
[L63] \\Rightarrow - \\frac{A}{2} \\cdot (\\rho_{\\text{eg}} \\cdot g) \\cdot \\frac{h^2}{3} - \\frac{A}{2} \\cdot (\\rho_{\\text{eg}} \\cdot g) \\cdot h \\cdot \\frac{2}{3} \\cdot b + \\frac{5}{4} \\cdot (\\rho_{\\text{eg}} \\cdot g) \\cdot h \\cdot \\frac{2}{3} \\cdot b^2 = 0
[L64] $$
[L65] 
[L66] $$
[L67] - \\frac{h^2}{6} - \\frac{b^2}{3} + \\frac{5}{6} \\cdot b^2 = 0 \\Rightarrow b^2 = \\frac{h^2}{3},\\quad b = \\frac{h}{\\sqrt{3}} \\approx 0.58 \\cdot h
[L68] $$
[L69] 
[L70] Pour le barrage de la Grande-Dixence : $h = 285 \\, \\text{m} \\Rightarrow b \\approx 0.7 \\cdot h$, $b = 200$
[L71] 
[L72] ---
[L73] 
[L74] ## Exercice 3.10.10
[L75] 
[L76] ![Système tronqué avec des forces distribuées réduites aux forces concentrées]
[L77] 
[L78] $$
[L79] \\frac{q_{1}}{\\ell_{2}} = \\frac{500}{12} = 33.6 \\, \\text{kN}
[L80] $$
[L81] 
[L82] Point d'application $P_{1}(5 \\, \\text{m} / 12 \\, \\text{m})$
[L83] 
[L84] $$
[L85] q_{\\text{tot}} = q \\cdot 8 \\, \\text{m} = 20 \\, \\frac{\\text{kN}}{\\text{m}} \\cdot 8 \\, \\text{m} = 160 \\, \\text{kN}
[L86] $$
[L87] 
[L88] $$
[L89] q_{2} = q \\cdot 3 \\, \\text{m} = 20 \\, \\frac{\\text{kN}}{\\text{m}} \\cdot 3 \\, \\text{m} = 60 \\, \\text{kN}
[L90] $$
[L91] 
[L92] Béton armé : $\\rho_{\\text{beton}} = 25 \\, \\frac{\\text{kN}}{\\text{m}^3}$
[L93] 
[L94] Pour calculer le poids propre : Considérer deux surfaces : un rectangle et un triangle
[L95] 
[L96] Rectangle :  
[L97] $$
[L98] G_{1} = \\rho_{\\text{beton}} \\cdot t \\cdot b \\cdot h = 25 \\, \\frac{\\text{kN}}{\\text{m}^3} \\cdot 1 \\, \\text{m} \\cdot 2 \\, \\text{m} \\cdot 8 \\, \\text{m} = 400 \\, \\text{kN}
[L99] $$  
[L100] Point d'application = (Centr. de la gravité du rectangle, $h_{1}(4 \\, \\text{m} / 4 \\, \\text{m})$)
[L101] 
[L102] Triangle :  
[L103] $$
[L104] G_{2} = \\frac{1}{2} \\rho_{\\text{beton}} \\cdot t \\cdot b \\cdot h = \\frac{1}{2} \\cdot 25 \\, \\frac{\\text{kN}}{\\text{m}^3} \\cdot 1 \\, \\text{m} \\cdot 2 \\, \\text{m} \\cdot 8 \\, \\text{m} = 300 \\, \\text{kN}
[L105] $$  
[L106] Point d'application = (Centr. de la gravité du triangle, $h_{2}(2 \\, \\text{m} / 2.67 \\, \\text{m})$)
[L107] 
[L108] Réduction au point O :
[L109] 
[L110] $$
[L111] \\rightarrow F_{R x} = -332.6 + 160 = -193.6 \\, \\text{kN}
[L112] $$
[L113] 
[L114] $$
[L115] \\uparrow F_{R y} = -332.6 - 60 - 400 - 300 = -1193.6 \\, \\text{kN}
[L116] $$
[L117] 
[L118] $$
[L119] h_{R o} = \\frac{q_{1}}{\\ell_{2}} \\cdot 8 \\, \\text{m} - \\frac{q_{1}}{\\ell_{2}} \\cdot 5 \\, \\text{m} - G_{1} \\cdot 4 \\, \\text{m} - G_{2} \\cdot 2 \\, \\text{m} - G_{\\text{tot}} \\cdot 1.5 \\, \\text{m}
[L120] $$
[L121] 
[L122] $$
[L123] = -1869.1 \\, \\text{kN} \\cdot \\text{m}
[L124] $$
[L125] 
[L126] ![Diagramme des moments fléchissants](#)
[L127] 
[L128] ---
[L129] 
[L130] # École Polytechnique Fédérale de Lausanne EPFL  
[L131] ## Section de Génie Civil  
[L132] ### Laboratoire du Génie Parasismique et Dynamique des Structures EESD  
[L133] #### Statique I (CIVIL-124)
[L134] 
[L135] 
[L136] ## Réduction au point S:
[L137] 
[L138] S: Point d'intersection de la ligne d'action de la force résultante au niveau du sol → Trouver le point $S(x_s, 0)$ où le moment résultant $M(x_s, 0)$
[L139] 
[L140] $$
[L141] \\begin{aligned}
[L142] & \\mathrm{F}_{\\mathrm{Rx}} = -193.6 \\, \\mathrm{kN} \\\\
[L143] & \\mathrm{F}_{\\mathrm{Ry}} = -1113.6 \\, \\mathrm{kN} \\\\
[L144] & \\mathrm{M}_{\\mathrm{Re}} = -1869 + 1113.6 \\cdot x_{\\mathrm{s}} = 0 \\quad \\Rightarrow \\quad x_{\\mathrm{s}} = 1.68 \\, \\mathrm{m}
[L145] \\end{aligned}
[L146] $$
[L147] 
[L148] ![Diagram showing forces and moments](#)
[L149] 
[L150] Les 4 figures suivantes sont équivalentes:
[L151] 
[L152] ![Four equivalent force diagrams](#)
[L153] 
[L154] ### Exercice 3.10.12 (Dr. Studer)
[L155] 
[L156] Réduction de R force à l'origine
[L157] 
[L158] ![Force reduction diagram](#)
[L159] 
[L160] $$
[L161] \\begin{aligned}
[L162] & \\mathrm{F}_{\\mathrm{x}} = 5 \\, \\mathrm{kN} \\\\
[L163] & \\mathrm{F}_{\\mathrm{y}} = 5 \\, \\mathrm{kN}
[L164] \\end{aligned}
[L165] $$
[L166] 
[L167] $$
[L168] \\begin{aligned}
[L169] & \\mathrm{F}_{\\mathrm{R}} = \\left\\{
[L170] \\begin{array}{l}
[L171] \\mathrm{F}_{\\mathrm{Rx}} = -\\mathrm{F}_{1} + \\mathrm{F}_{2} = 2 \\, \\mathrm{kN} \\\\
[L172] \\mathrm{F}_{\\mathrm{Ry}} = 0 \\\\
[L173] \\mathrm{F}_{\\mathrm{Rz}} = 0
[L174] \\end{array}
[L175] \\right. \\\\
[L176] & \\mathrm{M}_{\\mathrm{R}} = \\left\\{
[L177] \\begin{array}{l}
[L178] \\mathrm{M}_{\\mathrm{Rx}} = 0 \\\\
[L179] \\mathrm{M}_{\\mathrm{Ry}} = 0.6 \\mathrm{F}_{1} + 0.6 \\mathrm{F}_{2} = 4.8 \\, \\mathrm{kN} \\cdot \\mathrm{m} \\\\
[L180] \\mathrm{M}_{\\mathrm{Rz}} = 0.5 \\mathrm{F}_{1} - 0.5 \\mathrm{F}_{2} = -3 \\, \\mathrm{kN} \\cdot \\mathrm{m}
[L181] \\end{array}
[L182] \\right.
[L183] \\end{aligned}
[L184] $$
[L185] 
[L186] ---
[L187] 
[L188] Exercice 3.10.12
[L189] 
[L190] ![Diagram showing forces and moments acting on a structure]
[L191] 
[L192] $$
[L193] \\begin{aligned}
[L194] F_{Rx} &= \\Sigma F_x = 5 - 3 = 2 \\, \\text{[kN]} \\\\
[L195] F_{Ry} &= \\Sigma F_y = 0 \\, \\text{[kN]} \\\\
[L196] F_{Rz} &= Z F_z = 0 \\, \\text{[kN]} \\\\
[L197] M_{Rx} &= Z(yF_y - zF_z) + Z R_x^* = 0 \\, \\text{[kNm]} \\\\
[L198] M_{Ry} &= Z(zF_z - xF_x) + Z R_y^* = 0.6 \\cdot (-0.6 \\cdot 5) + 0.8 \\, \\text{[kNm]} \\\\
[L199] M_{Rz} &= Z(xF_y - yF_x) + Z R_z^* = -(0.9 \\cdot 5 + 0.6 \\cdot (-3)) = -2.0 \\, \\text{[kNm]}
[L200] \\end{aligned}
[L201] $$
[L202] 
[L203] $$
[L204] \\overrightarrow{F_R} = \\left\\{
[L205] \\begin{array}{c}
[L206] 2 \\\\
[L207] 0 \\\\
[L208] -2.0
[L209] \\end{array}
[L210] \\right\\} \\text{ [kN]}
[L211] $$
[L212] 
[L213] $$
[L214] \\overrightarrow{M_R} = \\left\\{
[L215] \\begin{array}{c}
[L216] 0 \\\\
[L217] 0 \\\\
[L218] -2.0
[L219] \\end{array}
[L220] \\right\\} \\text{ [kNm]}
[L221] $$
[L222] 
[L223] ---
[L224] 
[L225] # École Polytechnique Fédérale de Lausanne EPFL  
[L226] ## Section de Génie Civil  
[L227] ### Laboratoire du Génie Parasismique et Dynamique des Structures EESD  
[L228] #### Statique I (CIVIL-124)
[L229] 
[L230] ### Exercice 3.10.17
[L231] 
[L232] ![Diagram showing forces and angles in a truss structure]
[L233] 
[L234] $$
[L235] \\begin{aligned}
[L236] A_x &= 50 \\, \\text{kN} & A_y &= 0 \\, \\text{kN} \\\\
[L237] B_x &= 20 \\cdot \\cos 60^\\circ = 10 \\, \\text{kN} & B_y &= 20 \\cdot \\sin 60^\\circ = 17.32 \\, \\text{kN} \\\\
[L238] C_x &= C_y = C \\cdot \\frac{\\lambda}{\\sqrt{2}} \\\\
[L239] D_x &= D_3 & D_y &= 0
[L240] \\end{aligned}
[L241] $$
[L242] 
[L243] $$
[L244] \\begin{aligned}
[L245] \\Sigma F_{Rx} &= 0 & -A_x + B_x + C \\cdot \\frac{\\lambda}{\\sqrt{2}} + D &= 0 \\\\
[L246] \\Sigma F_{Ry} &= 0 & -B_y + C \\cdot \\frac{\\lambda}{\\sqrt{2}} &= 0 \\\\
[L247] C \\cdot B_y \\cdot \\sqrt{2} &= A \\cdot 32 \\cdot \\sqrt{2} = 24.5 \\, \\text{kN} \\\\
[L248] D &= A_x - B_x - C \\cdot \\frac{\\lambda}{\\sqrt{2}} \\\\
[L249] &= 50 - 10 - 24.5 \\cdot \\frac{\\lambda}{\\sqrt{2}} = 22.4 \\, \\text{kN}
[L250] \\end{aligned}
[L251] $$
[L252] 
[L253] ---
[L254] 
[L255] ## Exercice 3.10.18
[L256] 
[L257] ![Diagram showing equilibrium in the plane with forces Ax, Bx, By, and Q](#)
[L258] 
[L259] $$
[L260] \\begin{aligned}
[L261] & \\sum F_{x} = 0 \\Rightarrow A_{x} + B_{x} = 0 \\Rightarrow A_{x} = -B_{x} \\quad (1) \\\\
[L262] & \\sum F_{y} = 0 \\Rightarrow B_{y} - Q = 0 \\Rightarrow B_{y} = Q = 30 \\, \\text{kN} \\quad (2) \\\\
[L263] & \\sum M_{A} = 0 \\Rightarrow A_{x} \\cdot 3 \\, \\text{m} - Q \\cdot 3 \\, \\text{m} = 0 \\Rightarrow A_{x} = -30 \\, \\text{kN} \\quad (3)
[L264] \\end{aligned}
[L265] $$
[L266] 
[L267] $$
[L268] \\therefore B_{x} = +30 \\, \\text{kN}
[L269] $$
[L270] 
[L271] Si on désigne les forces $A_{x}, B_{x}$ et $B_{y}$ avec leur signe réel on a le système suivant:
[L272] 
[L273] ![Diagram showing forces Q and Q with angle alpha](#)
[L274] 
[L275] ---
[L276] 
[L277] ## Exercice 3.10.20
[L278] 
[L279] Exercice 3.40.20
[L280] 
[L281] Equivalence d'un système de force à un couple.
[L282] 
[L283] Exemple si :  
[L284] $$
[L285] \\left\\{
[L286] \\begin{array}{l}
[L287] R_{x} + R_{y} = 0 \\\\
[L288] M \\neq 0 \\text{ pas nappant à tout point}
[L289] \\end{array}
[L290] \\right.
[L291] $$
[L292] 
[L293] $$
[L294] \\left\\{
[L295] \\begin{array}{l}
[L296] R_{x} = 0 \\\\
[L297] R_{y} + 6 \\cdot 10 + 20 \\cdot 10 - 6 = 0 \\, \\text{kN} \\\\
[L298] M_{z} = -6 \\cdot 4 + 6 \\cdot 2 - 10 \\cdot 2 - 6 \\cdot 4 = -48 \\, \\text{kN} \\cdot \\text{m}
[L299] \\end{array}
[L300] \\right.
[L301] $$
[L302] 
[L303] On vérifie qu'en tout point $M_{R} = -48 \\, \\text{kN} \\cdot \\text{m} \neq 0$
[L304] 
[L305] ![Diagram showing forces at points B, C, D, E, and F with distances and angles](#)
[L306] 
[L307] Résultante de $F_{c}$ et $F_{a}$
[L308] 
[L309] Équilibre:  
[L310] $$
[L311] \\left\\{
[L312] \\begin{array}{l}
[L313] \\sum F_{x} = F_{c x} + F_{a x} = 0 \\rightarrow F_{a x} = F_{c} \\quad (F_{c} \\text{ horizontale}) \\\\
[L314] \\sum F_{y} = 0 \\rightarrow F_{a y} = 0 \\rightarrow F_{a} \\text{ horizontale } \\rightarrow F_{a} = F_{c} = F \\\\
[L315] \\sum M_{C} = 0 \\rightarrow M_{R} + 6 \\cdot F = 0 \\rightarrow F = 8 \\, \\text{kN} \\quad \\text{Les 2 forces } F_{c} \\text{ et } F_{a} \\text{ donnent} \\\\
[L316] \\text{un nouveau couple qui équivaut } M_{R}
[L317] \\end{array}
[L318] \\right.
[L319] $$
[L320] 
[L321] $$
[L322] \\mathbf{M}_{D} = \\text{ moment correspondant au couple des deux forces étendues seulement!}
[L323] $$
[L324] 
[L325] $$
[L326] \\sum m_{D} = -6 \\cdot 4 + 10 \\cdot 2 - 10 \\cdot 2 - 6 \\cdot 4 + F_{A X} \\cdot 6 + F_{A Y} \\cdot 6 = 0
[L327] $$
[L328] 
[L329] ---
[L330] 
[L331] ## Exercice additionnel 1 :
[L332] 
[L333] Résultats :  
[L334] $$
[L335] A_x = \\frac{Q}{\\sqrt{2}} \\left( \\leftarrow \\right),\\quad A_y = \\frac{Q}{4} \\left(\\sqrt{2} + 1\\right) \\left( \\uparrow \\right),\\quad B_y = \\frac{Q}{4} \\left(\\sqrt{2} + 3\\right) \\left( \\uparrow \\right)
[L336] $$
[L337] 
[L338] ![Structural diagram with forces and moments](#)
[L339] 
[L340] $$
[L341] \\begin{aligned}
[L342] & \\sum F_x = 0 \\\\
[L343] & \\sum F_y = 0 \\\\
[L344] & \\sum M_A = 0 \\\\
[L345] & \\sum M_B = 0
[L346] \\end{aligned}
[L347] $$
[L348] 
[L349] $$
[L350] \\begin{aligned}
[L351] & R_x + \\frac{Q}{\\sqrt{2}} = 0 & R_x = -\\frac{Q}{\\sqrt{2}} \\\\
[L352] & -Q \\cdot \\frac{L}{2} - Q \\cdot \\frac{3}{2} L + R_y \\cdot 2L = 0 & R_y = \\frac{Q}{2} \\left(L \\sqrt{2} + 3\\right) \\\\
[L353] & R_y - \\frac{Q}{\\sqrt{2}} - Q + R_y = 0 & R_y = \\frac{Q}{2} \\left(L \\sqrt{2} + 1\\right)
[L354] \\end{aligned}
[L355] $$
[L356]
```

you should output
```
[
    {"start_line": 10, "end_line": 71, "exercise_number": "additionnel 2", "is_solution": true},
    {"start_line": 74, "end_line": 129, "exercise_number": "3.10.10", "is_solution": true},
    {"start_line": 136, "end_line": 153, "exercise_number": "3.10.10", "is_solution": true},
    {"start_line": 154, "end_line": 187, "exercise_number": "3.10.12", "is_solution": true},
    {"start_line": 188, "end_line": 224, "exercise_number": "3.10.12", "is_solution": true},
    {"start_line": 230, "end_line": 254, "exercise_number": "3.10.17", "is_solution": true},
    {"start_line": 255, "end_line": 276, "exercise_number": "3.10.18", "is_solution": true},
    {"start_line": 277, "end_line": 330, "exercise_number": "3.10.20", "is_solution": true},
    {"start_line": 331, "end_line": 356, "exercise_number": "additionnel 2", "is_solution": true}
]
``` 
  
"""

    # Prepare response format
    class Exercise(BaseModel):
        start_line: int = Field(..., description="The first line of the Markdown chunk. Its contents are included in the chunk.")
        end_line: int = Field(..., description="The last line of the Markdown chunk. Its contents are included in the chunk.")
        number: str = Field(..., description="The exercise number, as referenced in the Markdown chunk. Typically one or more integers, as in `3` or `2.7`.")
        is_solution: bool = Field(..., description="Whether the Markdown chunk contains the solution of the exercise, as opposed to only the question.")

    class ExerciseList(BaseModel):
        exercises: List[Exercise]

    # Read Markdown file to be split
    md_text = md_path.read_text(encoding='utf-8')

    # Split by most common heading then try to merge as much as possible not exceeding the max lines
    max_lines = 2000
    md_texts = split_by_most_common_heading(md_text)
    md_texts = merge_up_to_max_lines(md_texts, max_lines)

    if len(md_texts) >= 2:
        all_n_lines = [md_text.splitlines() for md_text in md_texts]
        logging.info(f"Splitting {md_path} into {len(md_texts)} chunks (of {all_n_lines} lines) to extract exercises.")

    all_snippets = {}
    for md_text in md_texts:
        # Split into lines
        md_lines = md_text.splitlines()

        # Annotate Markdown with line numbers (ensuring empty line at the end)
        annotated_md_text = "\n".join(
            [f"[L{i}] {line}" for i, line in enumerate(md_lines, start=1)]
            + [""]
        )

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": annotated_md_text},
        ]

        # Call LLM to split into exercises
        exercise_list = send_llm_request(CONFIG['RCP_BASE_MODEL'], messages, response_format=ExerciseList)

        # Skip if no exercises
        if not exercise_list.exercises:
            continue

        # Retrieve snippets from lines
        snippets = {}
        for exercise in exercise_list.exercises:
            # Skip if lines make no sense
            lines_make_sense = 1 <= exercise.start_line <= exercise.end_line <= len(md_lines)
            if not lines_make_sense:
                logging.warning(f"While splitting {md_path} got exercise lines out of bounds: start {exercise.start_line}, end {exercise.end_line}, total {len(md_lines)}. Skipping...")
                continue

            # Fetch snippet from original document
            snippet = "\n".join(md_lines[exercise.start_line - 1: exercise.end_line])

            # Store snippet in object
            if (exercise.number, exercise.is_solution) in snippets:
                snippets[(exercise.number, exercise.is_solution)] += snippet
            else:
                snippets[(exercise.number, exercise.is_solution)] = snippet

        # Merge snippets
        all_snippets = all_snippets | snippets

    # Exercises could be repeated (statement and solution). Make unique by number by prioritising the solution
    all_snippets = {
        (number, is_solution): value
        for (number, is_solution), value in all_snippets.items()
        if is_solution or not all_snippets.get((number, True))
    }

    # Store exercises as individual Markdown files
    exercises_path.mkdir(parents=True, exist_ok=True)
    for (number, is_solution) in all_snippets:
        exercise_path = exercises_path / f"{number}.md"
        exercise_path.write_text(all_snippets[(number, is_solution)], encoding="utf-8")
