import os

# "Anthracnose", "Bacterial Blight", "Blight Borer", "Healthy", "Rot", "Fusarium Wilt"

management_dict = {
    0 : [
        "Select Haste or Ambe bahar varities. ",
        "Wider tree spacing, yearly pruning of trees. ",
        "Proper disposal of diseased leaves and twigs. "],
    1 : [
        "Wide row spacing. ", 
        "Selection of disease free seedlings for fresh planting. ", 
        "Pruning affected branches, fruits regularly and burning. ", 
        "Bahar should be done in Haste or Ambe Bahar. ", 
        "Give minimum four month rest after harvesting the fruits. "],
    2 : [
        "Clean cultivation and aintenance of health and vigour of the tree should be followe. ", 
        "The fruits if screened with polythene or paper bags may escape infestation. ", 
        "Removal and destruction of all the affected fruits."],
    3 : [
        "The plant is healthy. "],
    4 : [
        "Affected fruits should be collected and destroyed. ", 
        "Select Haste or Ambe bahar. ", 
        "Wider plant spacing, yearly pruning of trees. "],
    5 : [
        "Do not allow water to stagnate, try to create drainage facility. ", 
        "Do not irrigate for 2-3 days after drenching. "]
} 

treatment_dict = {
    0 : [
        "Carbendazim or Thiophanate methyl at 0.25ml/lit sprays. ",
        "Kitazin 48% EC @ 0.20% or 80ml in 80 l of water as required depending upon crop stage and plant protection equipment used."],
    1 : [
        "Before pruning it should be sprayed with 1% Bordeaux mixture. ", 
        "After Ethrel spraying Paste or smear with 0.5g Streptomycin Sulphate + 2.5g Copper oxy chloride + 200g red oxide per lit of water. ", 
        "Spray 0.5g Streptomycin Sulphate or Bacterinashak + 2.5g copper oxy chloride per liter of water. ", 
        "Next Day or another day spray with 1g ZnSo4 + 1g MgSo4 + 1g Boron + 1g CaSo4 Per litre of water. "],
    2 : [
        "Spray Dimethoate 0.06% prior to flowering is important. ", 
        "In severe condition spray methyl oxy-demeton 0.05% and repeat after fruit set. ", 
        "Spraying of Melathion 1ml/lit. "],
    3 : [
        "The plant is healthy. "],
    4 : [
        "Spraying Mancozeb (0.25%) or Captaf (0.25%) effectively control the disease. "],
    5 : [
        "At initial stage drench 2ml Propiconazole + 4ml Chloropyriphos per liter water solution, drench 8-10 lit of solution per tree. Drench with Formaldehyde @ 25 ml/l. "]
}

type_of_disease = ['Bacterial Blight / Telya', 'Anthracnose', 'Fruit Spot / Rot', 'Fusarium Wilt', 'Fruit Borer / Blight Blora']

disease_management_list = [
    [
        "Wide row spacing. ", 
        "Selection of disease free seedlings for fresh planting. ", 
        "Pruning affected branches, fruits regularly and burning. ", 
        "Bahar should be done in Haste or Ambe Bahar. ", 
        "Give minimum four month rest after harvesting the fruits. "
    ],
    [
        "Select Haste or Ambe bahar varities. ",
        "Wider tree spacing, yearly pruning of trees. ",
        "Proper disposal of diseased leaves and twigs. "
    ],
    [
        "Affected fruits should be collected and destroyed. ", 
        "Select Haste or Ambe bahar. ", 
        "Wider plant spacing, yearly pruning of trees. "
    ],
    [
        "Do not allow water to stagnate, try to create drainage facility. ", 
        "Do not irrigate for 2-3 days after drenching. "
    ],
    [
        "Clean cultivation and aintenance of health and vigour of the tree should be followe. ", 
        "The fruits if screened with polythene or paper bags may escape infestation. ", 
        "Removal and destruction of all the affected fruits."
    ]   
]

disease_treatment_list = [
    [
        "Before pruning it should be sprayed with 1% Bordeaux mixture. ", 
        "After Ethrel spraying Paste or smear with 0.5g Streptomycin Sulphate + 2.5g Copper oxy chloride + 200g red oxide per lit of water. ", 
        "Spray 0.5g Streptomycin Sulphate or Bacterinashak + 2.5g copper oxy chloride per liter of water. ", 
        "Next Day or another day spray with 1g ZnSo4 + 1g MgSo4 + 1g Boron + 1g CaSo4 Per litre of water. "
    ],
    [
        "Carbendazim or Thiophanate methyl at 0.25ml/lit sprays. ",
        "Kitazin 48% EC @ 0.20% or 80ml in 80 l of water as required depending upon crop stage and plant protection equipment used."
    ],
    [
        "Spraying Mancozeb (0.25%) or Captaf (0.25%) effectively control the disease. "
    ],
    [
        "At initial stage drench 2ml Propiconazole + 4ml Chloropyriphos per liter water solution, drench 8-10 lit of solution per tree. Drench with Formaldehyde @ 25 ml/l. "
    ],
    [
        "Spray Dimethoate 0.06% prior to flowering is important. ", 
        "In severe condition spray methyl oxy-demeton 0.05% and repeat after fruit set. ", 
        "Spraying of Melathion 1ml/lit. "
    ]
]


ROOT_PATH = os.getcwd()
DOCS_DIR = "static\Docs"
MULTI_MODEL_FILE = "MultiLabelModel_result.csv"
MULTI_MODEL_FILE_PATH = os.path.join(ROOT_PATH, DOCS_DIR, MULTI_MODEL_FILE)

MODELS_DIR = "static\Models"
MODELS_FILE_PATH = os.path.join(ROOT_PATH, MODELS_DIR)
