"""
Auto-generate JOA clinical labels from BERTopic top-keywords using a small
rule-based mapping of arthroplasty subspecialty keywords -> human-readable
clinical label. Domain-curated for the Journal of Arthroplasty.

Output: outputs/tables/clinical_labels_JOA.json mapping topic_id -> label.
"""
import pandas as pd
import json
import re
from pathlib import Path

TBL_DIR = Path("outputs/tables")
JOURNAL = "JOA"


# Lower-case keyword -> phrase template. First-matching wins. Order matters.
RULES = [
    # ---- INFECTION ----
    (("periprosthetic", "infection"),    "Periprosthetic Joint Infection Diagnosis and Treatment"),
    (("pji",),                            "Periprosthetic Joint Infection Diagnosis and Treatment"),
    (("infection", "synovial", "fluid"), "Synovial Fluid Biomarkers for PJI Diagnosis"),
    (("two", "stage", "exchange"),       "Two-Stage Exchange Arthroplasty for Infection"),
    (("debridement", "irridation"),      "Debridement, Antibiotics, and Implant Retention (DAIR)"),
    (("staphylococcus",),                 "Periprosthetic Infection Microbiology"),
    (("antibiotic", "cement"),           "Antibiotic-Loaded Cement Spacers"),
    (("biofilm",),                        "Implant Biofilm and Infection Biology"),

    # ---- ROBOTICS / NAVIGATION ----
    (("robotic", "assisted"),            "Robotic-Assisted Total Joint Arthroplasty"),
    (("robotic",),                        "Robotic-Assisted Total Joint Arthroplasty"),
    (("navigation", "computer"),         "Computer-Navigated Knee and Hip Arthroplasty"),
    (("navigated", "tka"),               "Computer-Navigated Knee Arthroplasty"),
    (("patient", "specific", "instrument"), "Patient-Specific Instrumentation for TKA"),
    (("augmented", "reality"),           "Augmented Reality and Mixed-Reality Surgical Guidance"),

    # ---- OUTPATIENT / ENHANCED RECOVERY ----
    (("outpatient", "arthroplasty"),     "Outpatient and Same-Day Discharge Arthroplasty"),
    (("same", "day", "discharge"),       "Outpatient and Same-Day Discharge Arthroplasty"),
    (("ambulatory", "surgery"),          "Ambulatory Surgery Center Joint Replacement"),
    (("rapid", "recovery"),              "Rapid Recovery and Enhanced Recovery Protocols"),
    (("enhanced", "recovery"),           "Enhanced Recovery After Surgery (ERAS) Pathways"),
    (("length", "of", "stay"),           "Length of Stay and Discharge Disposition"),
    (("discharge", "home"),              "Discharge Disposition and Skilled Nursing Use"),

    # ---- PAIN MANAGEMENT ----
    (("opioid",),                         "Opioid Use, Prescription Patterns, and Stewardship"),
    (("multimodal", "analgesia"),        "Multimodal Analgesia After Joint Arthroplasty"),
    (("periarticular", "injection"),     "Periarticular Injection for Postoperative Pain"),
    (("adductor", "canal"),              "Adductor Canal and Peripheral Nerve Blocks"),
    (("femoral", "nerve", "block"),      "Femoral Nerve Block in Knee Arthroplasty"),
    (("tranexamic", "acid"),             "Tranexamic Acid and Perioperative Blood Loss"),

    # ---- VTE / DVT / BLEEDING ----
    (("venous", "thromboembolism"),      "VTE Prophylaxis After Total Joint Arthroplasty"),
    (("dvt", "prophylaxis"),             "DVT Prophylaxis Regimens"),
    (("aspirin", "vte"),                  "Aspirin vs Anticoagulant VTE Prophylaxis"),
    (("aspirin",),                        "Aspirin for VTE Prophylaxis"),
    (("rivaroxaban",),                    "Direct Oral Anticoagulants After Arthroplasty"),
    (("enoxaparin",),                     "Low-Molecular-Weight Heparin Prophylaxis"),
    (("warfarin",),                       "Warfarin Anticoagulation After Arthroplasty"),
    (("blood", "transfusion"),           "Blood Loss and Transfusion Management"),

    # ---- BEARING SURFACES / WEAR ----
    (("highly", "crosslinked"),          "Highly Crosslinked Polyethylene Wear"),
    (("polyethylene", "wear"),           "Polyethylene Wear and Osteolysis"),
    (("polyethylene", "liner"),          "Polyethylene Liner Performance"),
    (("metal", "on", "metal"),           "Metal-on-Metal Bearings and Adverse Reactions"),
    (("ceramic",),                        "Ceramic-on-Ceramic Bearings and Fracture"),
    (("osteolysis",),                     "Osteolysis and Implant Loosening"),
    (("wear", "particles"),              "Wear Particle Biology and Macrophage Response"),

    # ---- IMPLANT DESIGN ----
    (("cementless", "femoral"),          "Cementless Femoral Stem Fixation"),
    (("cemented", "femoral"),            "Cemented Femoral Stem Fixation"),
    (("modular", "neck"),                "Modular Femoral Necks and Taper Corrosion"),
    (("dual", "mobility"),               "Dual-Mobility Acetabular Constructs"),
    (("monoblock",),                      "Monoblock Acetabular Cup Design"),
    (("short", "stem"),                  "Short-Stem Femoral Components"),
    (("tapered", "stem"),                "Tapered Femoral Stems"),
    (("press", "fit"),                   "Press-Fit Acetabular Component Fixation"),
    (("highly", "porous"),               "Highly Porous Acetabular Components"),

    # ---- HIP-SPECIFIC ----
    (("anterior", "approach"),           "Direct Anterior Approach Total Hip Arthroplasty"),
    (("posterior", "approach"),          "Posterior Approach Total Hip Arthroplasty"),
    (("dislocation", "tha"),              "Hip Dislocation and Instability After THA"),
    (("hip", "dislocation"),             "Hip Dislocation and Instability"),
    (("hip", "fracture"),                "Hip Fracture and Hemiarthroplasty"),
    (("femoral", "neck", "fracture"),    "Femoral Neck Fracture Hemiarthroplasty"),
    (("hip", "resurfacing"),             "Hip Resurfacing Arthroplasty"),
    (("acetabular", "cup"),              "Acetabular Cup Positioning and Fixation"),
    (("offset", "leg"),                   "Femoral Offset and Leg Length"),

    # ---- KNEE-SPECIFIC ----
    (("uka", "unicompartmental"),        "Unicompartmental Knee Arthroplasty"),
    (("unicompartmental",),               "Unicompartmental Knee Arthroplasty"),
    (("posterior", "stabilized"),         "Posterior-Stabilized vs CR Knee Designs"),
    (("cruciate", "retaining"),          "Cruciate-Retaining Knee Arthroplasty"),
    (("medial", "pivot"),                "Medial-Pivot Knee Designs"),
    (("kinematic", "alignment"),         "Kinematic Alignment in Knee Arthroplasty"),
    (("mechanical", "alignment"),        "Mechanical Alignment in Knee Arthroplasty"),
    (("patellar", "resurfacing"),        "Patellar Resurfacing in TKA"),
    (("flexion", "contracture"),         "Knee Flexion and Range of Motion"),
    (("manipulation", "under", "anesthesia"), "Manipulation Under Anesthesia for Stiff Knee"),
    (("tibial", "component"),            "Tibial Component Alignment and Fixation"),
    (("femoral", "component"),           "Femoral Component Rotation and Sizing"),

    # ---- COMPLICATIONS ----
    (("periprosthetic", "fracture"),     "Periprosthetic Fracture Management"),
    (("revision", "tha"),                "Revision Total Hip Arthroplasty"),
    (("revision", "tka"),                "Revision Total Knee Arthroplasty"),
    (("revision", "arthroplasty"),       "Revision Joint Arthroplasty Outcomes"),
    (("aseptic", "loosening"),           "Aseptic Loosening of Hip and Knee Implants"),
    (("readmission",),                    "30-Day Readmission and Postoperative Complications"),
    (("mortality", "after"),              "Mortality and Major Complications After Arthroplasty"),

    # ---- DATA SCIENCE / OUTCOMES ----
    (("machine", "learning"),            "Machine Learning Prediction of Arthroplasty Outcomes"),
    (("artificial", "intelligence"),     "Artificial Intelligence in Joint Replacement"),
    (("registry", "arthroplasty"),       "National and International Arthroplasty Registries"),
    (("national", "joint", "registry"),   "National Joint Replacement Registry Studies"),
    (("medicare",),                       "Medicare Reimbursement and Bundled Payments"),
    (("bundled", "payment"),             "Bundled Payment and Value-Based Care"),
    (("cost", "effectiveness"),          "Cost-Effectiveness of Joint Arthroplasty"),
    (("value", "based"),                 "Value-Based Care Models in Arthroplasty"),

    # ---- PROMs ----
    (("patient", "reported", "outcome"), "Patient-Reported Outcome Measures (PROMs)"),
    (("koos",),                           "KOOS and Knee-Specific PROMs"),
    (("hoos",),                           "HOOS and Hip-Specific PROMs"),
    (("promis",),                         "PROMIS Outcome Measurement"),
    (("oxford", "knee"),                 "Oxford Knee Score and Functional Outcomes"),
    (("womac",),                          "WOMAC Functional Assessment"),
    (("satisfaction",),                   "Patient Satisfaction After Joint Arthroplasty"),

    # ---- COMORBIDITIES / RISK FACTORS ----
    (("obesity", "bmi"),                  "Obesity and BMI as Risk Factor for Arthroplasty"),
    (("morbid", "obesity"),              "Morbid Obesity and Arthroplasty Complications"),
    (("diabetes",),                       "Diabetes Mellitus and Arthroplasty Outcomes"),
    (("smoking",),                        "Smoking and Joint Arthroplasty Risk"),
    (("hemoglobin", "a1c"),              "HbA1c and Surgical Outcome Optimization"),
    (("malnutrition", "albumin"),         "Nutritional Status and Surgical Risk"),
    (("frailty",),                        "Frailty and Arthroplasty Outcomes"),
    (("opioid", "use"),                  "Preoperative Opioid Use and Outcomes"),
    (("mental", "health"),               "Depression and Mental Health in Arthroplasty"),

    # ---- DEMOGRAPHICS ----
    (("racial", "disparities"),          "Racial and Ethnic Disparities in Arthroplasty"),
    (("socioeconomic",),                  "Socioeconomic Determinants of Arthroplasty Outcomes"),
    (("gender", "differences"),          "Sex and Gender Differences in Joint Replacement"),
    (("young", "patients"),              "Joint Arthroplasty in Young Patients"),
    (("elderly", "octogenarian"),        "Arthroplasty in Octogenarian and Elderly Patients"),

    # ---- TUMOR / ONCOLOGY ----
    (("megaprosthesis",),                 "Megaprosthesis Reconstruction"),
    (("tumor", "endoprosthesis"),         "Endoprosthetic Reconstruction for Tumors"),
    (("oncology", "limb"),                "Limb-Salvage Oncology Arthroplasty"),

    # ---- BASIC SCIENCE / TRIBOLOGY ----
    (("finite", "element"),              "Finite Element Analysis of Implant Biomechanics"),
    (("biomechanical", "cadaveric"),     "Biomechanical Cadaveric Testing"),
    (("retrieval", "analysis"),          "Implant Retrieval Analysis"),
    (("tribology",),                      "Tribology and Wear Testing of Implants"),

    # ---- MISC ----
    (("review", "literature"),           "Narrative Review Articles on Joint Arthroplasty"),
    (("systematic", "review"),           "Systematic Reviews and Meta-Analyses"),
    (("meta", "analysis"),                "Systematic Reviews and Meta-Analyses"),
    (("randomized", "controlled"),        "Randomized Controlled Trials in Arthroplasty"),
    (("case", "report"),                  "Case Reports of Unusual Arthroplasty Complications"),
    (("anesthesia", "spinal"),            "Spinal vs General Anesthesia for Arthroplasty"),
    (("regional", "anesthesia"),          "Regional Anesthesia for Joint Replacement"),
    (("physical", "therapy"),             "Postoperative Rehabilitation and Physical Therapy"),
    (("preoperative", "education"),       "Preoperative Education and Optimization"),
    (("telehealth",),                     "Telehealth and Remote Postoperative Monitoring"),
    (("covid",),                          "COVID-19 Pandemic Impact on Arthroplasty Practice"),
    (("simulation", "training"),          "Surgical Simulation and Resident Training"),
    (("learning", "curve"),               "Surgical Learning Curve in New Techniques"),
    (("vitamin", "d"),                    "Vitamin D and Bone Health in Arthroplasty"),
    (("bone", "loss"),                    "Bone Loss and Reconstruction in Revision"),

    # ---- ADDITIONAL ARTHROPLASTY-SPECIFIC (derived from JOA topic keywords) ----
    # Blood management / TXA
    (("txa", "blood"),                   "Tranexamic Acid and Perioperative Blood Management"),
    (("randomized", "txa"),              "Tranexamic Acid Randomized Controlled Trials"),
    (("randomized", "blood"),            "Perioperative Blood Management RCTs"),

    # Acetabular positioning / cup orientation
    (("anteversion", "degrees"),         "Acetabular Cup Orientation and Component Positioning"),
    (("anteversion", "navigation"),      "Acetabular Cup Orientation and Component Positioning"),
    (("anteversion", "axis"),            "Acetabular Cup Orientation and Component Positioning"),

    # Primary THA technique
    (("acetabular", "technique"),        "Primary Total Hip Arthroplasty Technique and Outcomes"),
    (("acetabular", "bone", "hips"),     "Primary Total Hip Arthroplasty Technique and Outcomes"),

    # Statistical methods / study design
    (("ci", "odds"),                     "Observational Study Design and Statistical Methods"),
    (("odds", "ratio"),                  "Observational Study Design and Statistical Methods"),

    # Kinematics / gap balancing
    (("flexion", "kinematics"),          "Knee Kinematics, Gap Balancing, and ROM"),
    (("posterior", "kinematics"),        "Knee Kinematics, Gap Balancing, and ROM"),
    (("gap", "balancing"),               "Gap Balancing in Total Knee Arthroplasty"),

    # THA long-term survivorship
    (("survivorship", "acetabular"),     "THA Survivorship and Long-Term Aseptic Loosening"),
    (("survivorship", "loosening"),      "THA Survivorship and Long-Term Aseptic Loosening"),

    # Implant retrieval / corrosion analysis
    (("retrieved", "wear"),              "Implant Retrieval and Corrosion Analysis"),
    (("retrieved", "damage"),            "Implant Retrieval and Corrosion Analysis"),
    (("retrieval", "corrosion"),         "Implant Retrieval and Corrosion Analysis"),

    # Metal ions / cobalt-chrome toxicity
    (("metal", "ion"),                   "Metal Ion Release and Cobalt-Chromium Toxicity"),
    (("cobalt", "chromium"),             "Metal Ion Release and Cobalt-Chromium Toxicity"),
    (("ion", "cobalt"),                  "Metal Ion Release and Cobalt-Chromium Toxicity"),
    (("metal", "levels"),                "Metal Ion Monitoring After Metal-on-Metal Arthroplasty"),

    # ALTR / adverse tissue reactions
    (("tissue", "corrosion"),            "Adverse Local Tissue Reactions and Implant Corrosion"),
    (("tissue", "metal"),                "Adverse Local Tissue Reactions and Implant Corrosion"),

    # AAHKS surgeon survey
    (("aahks", "survey"),               "AAHKS Member Survey Studies"),
    (("survey", "respondents"),          "Surgeon Survey Research in Arthroplasty"),
    (("members", "survey"),              "Surgeon Survey Research in Arthroplasty"),

    # Value-based care
    (("health", "care", "payment"),      "Value-Based Care and Alternative Payment Models"),
    (("value", "payment"),               "Value-Based Care and Alternative Payment Models"),

    # Biomechanical / fixation testing
    (("biomechanical", "fixation"),      "Biomechanical Testing and Implant Stability"),
    (("biomechanical", "stability"),     "Biomechanical Testing and Implant Stability"),
    (("fixation", "stability"),          "Implant Fixation and Biomechanical Stability"),

    # Survivorship / competing risks
    (("cumulative", "survivorship"),     "Implant Survival Analysis and Competing Risk Methods"),
    (("cumulative", "incidence"),        "Implant Survival Analysis and Competing Risk Methods"),

    # Taper / modular junction corrosion
    (("taper", "corrosion"),             "Femoral Head Taper Corrosion and Modular Junction Failure"),
    (("head", "taper"),                  "Femoral Head Taper Corrosion and Modular Junction Failure"),
    (("modular", "taper"),               "Femoral Head Taper Corrosion and Modular Junction Failure"),

    # UKA functional outcomes
    (("functional", "uka"),              "Unicompartmental Knee Arthroplasty Functional Outcomes"),
    (("alignment", "uka"),               "Unicompartmental Knee Arthroplasty Alignment Outcomes"),

    # Hip-spine syndrome
    (("iliopsoas", "lumbar"),            "Hip-Spine Syndrome and Lumbar Fusion After THA"),
    (("lumbar", "dislocation"),          "Lumbar Fusion and THA Dislocation Risk"),
    (("spinopelvic", "fusion"),          "Spinopelvic Parameters and Hip-Spine Conflict"),
    (("iliopsoas", "fusion"),            "Hip-Spine Syndrome and Spinopelvic Mechanics"),

    # Registry revision risk
    (("registry", "revision"),           "Registry-Based Revision Risk and Implant Survivorship"),
    (("hr", "revision"),                 "Hazard Ratio Analysis of Revision Risk"),

    # Periprosthetic fracture
    (("fracture", "orif"),               "Periprosthetic Fracture Treatment and Fixation"),
    (("periprosthetic", "fracture"),     "Periprosthetic Fracture Management"),
    (("eto", "fracture"),                "Extended Trochanteric Osteotomy and Periprosthetic Fracture"),
    (("intertrochanteric", "fracture"),  "Intertrochanteric Fracture and Hip Arthroplasty"),

    # Osteonecrosis femoral head
    (("onfh",),                          "Osteonecrosis of the Femoral Head"),
    (("collapse", "femoral"),            "Femoral Head Collapse and Osteonecrosis"),
    (("osteonecrosis",),                 "Osteonecrosis of the Femoral Head"),

    # Hip dysplasia
    (("dysplasia", "osteotomy"),         "Hip Dysplasia and Periacetabular Osteotomy"),
    (("dysplasia", "hips"),              "Developmental Hip Dysplasia and THA Technique"),
    (("shortening", "dysplasia"),        "Limb Shortening and Hip Dysplasia Correction"),

    # Online / social media / training survey
    (("online", "fellowship"),           "Online Research Quality and Surgical Training Surveys"),
    (("online", "search"),               "Internet-Based Patient Research and Online Information"),

    # Lower extremity alignment / ligament
    (("acl", "varus"),                   "Lower Extremity Deformity and Ligament Reconstruction"),
    (("ankle", "varus"),                 "Lower Extremity Alignment and Deformity Correction"),

    # Direct anterior approach THA
    (("daa", "anterior"),                "Direct Anterior Approach Total Hip Arthroplasty"),
    (("anterior", "da"),                 "Direct Anterior Approach Total Hip Arthroplasty"),

    # Radiographic implant survival
    (("survivorship", "radiographic"),   "Radiographic Implant Survival and Loosening Assessment"),
    (("survival", "radiographic"),       "Radiographic Implant Survival Assessment"),

    # Ambulatory surgery center / outpatient TJA
    (("outpatient", "asc"),              "Ambulatory Surgery Center and Outpatient TJA"),
    (("asc", "day"),                     "Ambulatory Surgery Center Arthroplasty"),

    # Discharge prediction models
    (("discharge", "predictive"),        "Machine Learning Discharge Prediction After TJA"),
    (("risk", "assessment", "discharge"), "Risk Assessment and Discharge Disposition Models"),

    # Minimally invasive approaches
    (("minimally", "invasive"),          "Minimally Invasive Surgical Approaches in TJA"),
    (("incision", "invasive"),           "Minimally Invasive and Mini-Incision Arthroplasty"),

    # Soft tissue balancing / knee
    (("release", "varus"),               "Soft Tissue Balancing and Ligament Release in TKA"),
    (("flexion", "varus"),               "Flexion-Extension Gap Balancing in TKA"),

    # OR contamination / implant squeaking
    (("contamination", "air"),           "Operating Room Air Contamination and Infection Prevention"),
    (("noise", "squeaking"),             "Implant Noise and Squeaking in TJA"),

    # Spinopelvic parameters / pelvic tilt
    (("spinopelvic", "seated"),          "Spinopelvic Parameters and Cup Positioning in THA"),
    (("pelvic", "standing"),             "Pelvic Tilt and Functional Acetabular Positioning"),
    (("standing", "pelvic"),             "Functional Alignment and Pelvic Kinematics in THA"),

    # Morbid obesity and TJA
    (("morbidly", "obese"),              "Morbid Obesity and Joint Arthroplasty Complications"),
    (("obese", "bmi"),                   "Obesity and BMI Effects on Arthroplasty Outcomes"),
    (("bariatric", "surgery"),           "Bariatric Surgery Prior to Joint Arthroplasty"),

    # Cementless stem subsidence
    (("subsidence", "cementless"),       "Cementless Stem Subsidence and Early Fixation"),
    (("subsidence", "survivorship"),     "Femoral Stem Subsidence and Long-Term Survivorship"),

    # TKA alignment
    (("varus", "axis"),                  "Tibial Mechanical Axis and Varus-Valgus Alignment"),
    (("alignment", "axis"),              "Component Alignment and Mechanical Axis in TKA"),

    # Ceramic osteolysis
    (("ceramic", "osteolysis"),          "Ceramic Components and Periprosthetic Osteolysis"),
    (("osteolysis", "stem"),             "Femoral Stem Osteolysis and Bone Remodeling"),

    # Same-day discharge
    (("sdd", "day"),                     "Same-Day Discharge Protocols in Joint Arthroplasty"),
    (("los", "sdd"),                     "Length of Stay and Same-Day Discharge Outcomes"),

    # Range of motion
    (("flexion", "rom"),                 "Knee Flexion and Range of Motion After TKA"),
    (("medial", "rom"),                  "ROM and Medial Release in Total Knee Arthroplasty"),

    # RVU / surgical work
    (("rvu", "work"),                    "Relative Value Units and Surgical Work in TJA"),
    (("rvu", "minutes"),                 "Operative Time, RVU, and Surgical Efficiency"),

    # Cardiac complications / ileus
    (("cardiac", "outpatient"),          "Cardiac Complications and Medical Comorbidities After TJA"),
    (("ileus", "cardiac"),               "Postoperative Ileus and Cardiac Events After TJA"),

    # Volume projections
    (("projected", "annual"),            "Arthroplasty Volume Trends and Future Projections"),
    (("procedures", "projected"),        "Projected Arthroplasty Demand and Healthcare Burden"),
    (("rtha",),                          "Revision Total Hip Arthroplasty Volume and Burden"),

    # Anatomy / neurovascular
    (("neurovascular", "nerve"),         "Neurovascular Anatomy and Complication Avoidance"),
    (("vastus", "nerve"),                "Vastus and Nerve Anatomy in Knee Arthroplasty"),

    # Femoral stem survival
    (("stem", "loosening"),              "Femoral Stem Survival and Aseptic Loosening"),
    (("stem", "survival"),               "Femoral Stem Long-Term Survival Studies"),

    # Perioperative antibiotic / VTE prophylaxis
    (("aki", "vancomycin"),              "Vancomycin Prophylaxis and Acute Kidney Injury Risk"),
    (("vte", "vancomycin"),              "Perioperative Antibiotic and VTE Prophylaxis Protocols"),

    # Deep learning
    (("deep", "learning"),               "Deep Learning and Computer Vision in Arthroplasty"),
    (("learning", "images"),             "Machine Learning Image Analysis in Joint Replacement"),

    # Bibliometrics / publication bias
    (("publication", "spin"),            "Research Quality, Spin, and Publication Bias in TJA"),
    (("journal", "articles"),            "Bibliometric Analysis of Joint Arthroplasty Literature"),
    (("spin", "published"),              "Research Spin and Reporting Quality in Arthroplasty"),
    (("fragility", "rcts"),              "Clinical Trial Quality and Fragility Index in Arthroplasty"),
    (("consensus", "iqr"),               "Consensus Statements and Delphi Methodology"),

    # Valgus / ankle alignment
    (("varus", "valgus"),                "Coronal Plane Alignment in Total Knee Arthroplasty"),
    (("alignment", "valgus"),            "Valgus Deformity and TKA Alignment Outcomes"),

    # Bilateral TKA
    (("btka", "bilateral"),              "Bilateral Total Knee Arthroplasty and Simultaneous Staging"),
    (("bilateral", "staged"),            "Staged vs Simultaneous Bilateral Knee Arthroplasty"),

    # KOOS simultaneous
    (("koos", "simultaneous"),           "KOOS Outcomes After Simultaneous Bilateral TKA"),

    # Remaining edge cases
    (("patellar", "cruciate"),           "Patellar Resurfacing, PCL Management, and Knee Design"),
    (("cruciate", "medial"),             "Posterior Cruciate Ligament and Medial Compartment in TKA"),
    (("publication", "authors"),         "Authorship Trends and Publication Patterns in Arthroplasty"),
    (("authors", "articles"),            "Authorship Trends and Publication Patterns in Arthroplasty"),
    (("knee", "society"),                "Knee Society Score and Clinical Outcome Reporting"),
    (("degrees", "society"),             "Knee Society Score and ROM Outcome Assessment"),
]


def label_topic(top_words: list[str]) -> str:
    txt = " ".join(top_words).lower()
    txt_tokens = set(re.findall(r"[a-z]+", txt))
    best_match = None
    best_hits  = 0
    for kw_tuple, label in RULES:
        hits = sum(1 for kw in kw_tuple if kw in txt_tokens)
        if hits == len(kw_tuple) and hits > best_hits:
            best_match = label
            best_hits  = hits
    return best_match


def main():
    ti = pd.read_csv(TBL_DIR / f"topic_info_{JOURNAL}.csv")
    ti = ti[ti["Topic"] != -1].copy()

    # Pull top words from topic_info Name column which is "0_word_word_..."
    labels = {}
    used   = {}
    for _, row in ti.iterrows():
        tid = int(row["Topic"])
        # Name has form "id_word1_word2_..." -- split off id
        name = str(row.get("Name", ""))
        parts = name.split("_")
        top_words = parts[1:] if len(parts) > 1 else parts
        label = label_topic(top_words)
        if label is None:
            # Fallback to "Topic N: word1 word2 word3" form
            label = "Other: " + " / ".join(top_words[:3]).title()
        # Disambiguate duplicates by appending the top-1 distinguishing word
        if label in used:
            distinguishers = [w for w in top_words if w not in label.lower().split()]
            extra = distinguishers[0].title() if distinguishers else str(tid)
            label = f"{label} ({extra})"
        used[label] = used.get(label, 0) + 1
        labels[tid] = label

    LABELS_JSON = TBL_DIR / f"clinical_labels_{JOURNAL}.json"
    LABELS_JSON.write_text(json.dumps(labels, indent=2))
    print(f"Wrote {len(labels)} labels to {LABELS_JSON}")
    for k, v in sorted(labels.items()):
        print(f"  Topic {k:>3}: {v}")


if __name__ == "__main__":
    main()
