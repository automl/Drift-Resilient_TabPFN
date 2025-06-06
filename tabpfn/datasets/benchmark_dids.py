# OpenML CC-18 Benchmark (Filtered by N_samples < 2000, N feats < 100, N classes < 10)
tabpfn_dids_classification = [
    11,
    14,
    15,
    16,
    18,
    22,
    23,
    29,
    31,
    37,
    50,
    54,
    188,
    458,
    469,
    1049,
    1050,
    1063,
    1068,
    1510,
    1494,
    1480,
    1462,
    1464,
    6332,
    23381,
    40966,
    40982,
    40994,
    40975,
]

openml_cc_18_classification = [
    "3@3",
    "6@6",
    "11@11",
    "12@12",
    "14@14",
    "15@15",
    "16@16",
    "18@18",
    "22@22",
    "23@23",
    "28@28",
    "29@29",
    "31@31",
    "32@32",
    "37@37",
    "43@44",
    "45@46",
    "49@50",
    "53@54",
    "219@151",
    "2074@182",
    "2079@188",
    "3021@38",
    "3022@307",
    "3481@300",
    "3549@458",
    "3560@469",
    "3573@554",
    "3902@1049",
    "3903@1050",
    "3904@1053",
    "3913@1063",
    "3917@1067",
    "3918@1068",
    "7592@1590",
    "9910@4134",
    "9946@1510",
    "9952@1489",
    "9957@1494",
    "9960@1497",
    "9964@1501",
    "9971@1480",
    "9976@1485",
    "9977@1486",
    "9978@1487",
    "9981@1468",
    "9985@1475",
    "10093@1462",
    "10101@1464",
    "14952@4534",
    "14954@6332",
    "14965@1461",
    "14969@4538",
    "14970@1478",
    "125920@23381",
    "125922@40499",  # Removed because its image data
    "146195@40668",
    "146800@40966",
    "146817@40982",
    "146819@40994",
    "146820@40983",
    "146821@40975",
    "146822@40984",
    "146824@40979",
    "146825@40996",
    "167119@41027",
    "167120@23517",
    # '167121@40923', # Removed because its image data
    # '167124@40927', # Removed because its image data
    "167125@40978",
    "167140@40670",
    "167141@40701",
]

# Grinzstjan Benchmark
grinzstjan_numerical_regression = [
    "361089@44025",
    "361090@44026",
    "361091@44027",
    "361072@44132",
    "361073@44133",  # overlaps with automl benchmark
    "361074@44134",  # overlaps with automl benchmark
    "361075@44135",
    "361076@44136",  # overlaps with automl benchmark
    "361077@44137",
    "361078@44138",
    "361079@44139",  # overlaps with automl benchmark
    "361080@44140",  # overlaps with automl benchmark
    "361081@44141",  # overlaps with automl benchmark
    "361082@44142",
    "361083@44143",
    "361084@44144",  # overlaps with automl benchmark
    "361085@44145",
    "361086@44146",
    "361087@44147",
    "361088@44148",
]

grinzstjan_numerical_regression_without_automl_overlap = [
    "361089@44025",
    "361090@44026",
    "361091@44027",
    "361072@44132",
    "361075@44135",
    "361077@44137",
    "361078@44138",
    "361082@44142",
    "361083@44143",
    "361085@44145",
    "361086@44146",
    "361087@44147",
    "361088@44148",
]

grinzstjan_categorical_regression = [
    "361093@44055",
    "361094@44056",
    "361096@44059",  # overlaps with automl benchmark
    "361097@44061",  # overlaps with automl benchmark
    "361098@44062",  # overlaps with automl benchmark
    "361099@44063",
    "361101@44065",  # overlaps with automl benchmark
    "361102@44066",  # overlaps with automl benchmark
    "361103@44068",
    "361104@44069",
    "361287@45041",  # overlaps with automl benchmark
    "361288@45042",  # overlaps with automl benchmark
    "361289@45043",
    "361291@45045",
    "361292@45046",  # overlaps with automl benchmark
    "361293@45047",  # overlaps with automl benchmark
    "361294@45048",
]
grinzstjan_categorical_regression_without_automl_overlap = [
    "361093@44055",
    "361094@44056",
    "361099@44063",
    "361103@44068",
    "361104@44069",
    "361289@45043",
    "361291@45045",
    "361294@45048",
]

grinzstjan_numerical_classification = [
    "361055@44089",
    "361060@44120",
    "361061@44121",
    "361062@44122",
    "361063@44123",
    "361065@44125",
    "361066@44126",
    "361068@44128",
    "361069@44129",
    "361070@44130",
    "361273@45022",
    "361274@45021",
    "361275@45020",
    "361276@45019",
    "361277@45028",
    "361278@45026",
]
grinzstjan_numerical_classification_without_automl_overlap = [
    "361055@44089",
    "361060@44120",
    "361062@44122",
    "361063@44123",
    "361065@44125",
    "361070@44130",
    "361275@45020",
    "361277@45028",
    "361278@45026",
]

grinzstjan_categorical_classification = [
    "361110@44156",
    "361111@44157",
    "361113@44159",
    "361114@44160",
    "361115@44161",
    "361116@44162",
    "361127@44186",
]
grinzstjan_categorical_classification_without_automl_overlap = [
    "361110@44156",  # too imbalanced for evaluation w/
    "361111@44157",
    "361114@44160",
    "361115@44161",
    "361116@44162",
]

# Automl Benchmark
automl_dids_classification = [
    "2073@181",
    "3945@1111",
    "7593@1596",
    "10090@1457",
    "146818@40981",
    "146820@40983",
    "167120@23517",
    "168350@1489",
    "168757@31",
    "168784@40982",
    "168868@41138",
    "168909@41163",
    "168910@41164",
    "168911@41143",
    "189354@1169",
    "189355@41167",
    "189356@41147",
    "189922@41158",
    "190137@1487",
    "190146@54",
    "190392@41144",
    "190410@41145",
    "190411@41156",
    "190412@41157",
    "211979@41168",
    "211986@4541",
    "359953@1515",
    "359954@188",
    "359955@1464",
    "359956@1494",
    "359957@1468",
    "359958@1049",
    "359959@23",
    "359960@40975",
    "359961@12",
    "359962@1067",
    "359963@40984",
    "359964@40670",
    "359965@3",
    "359966@40978",
    "359967@4134",
    "359968@40701",
    "359969@1475",
    "359970@4538",
    "359971@4534",
    "359972@41146",
    "359973@41142",
    "359974@40498",
    "359975@40900",
    # "359976@40996", # Removed because its image data
    "359977@40668",
    "359979@4135",
    "359980@1486",
    "359981@41027",
    "359982@1461",
    "359983@1590",
    "359984@41169",
    "359985@41166",
    "359986@41165",
    "359987@40685",
    "359988@41159",
    "359989@41161",
    "359990@41150",
    "359991@41162",
    "359992@42733",
    "359993@42734",
    "359994@42732",
    "360112@42746",
    "360113@42742",
    "360114@42769",
    "360975@43072",
]

openml_ctr23_regression = [
    "361234@44956",
    "361235@44957",
    "361236@44958",
    "361237@44959",
    "361241@44963",
    "361242@44964",
    "361243@44965",
    "361244@44966",
    "361247@44969",
    "361249@44971",
    "361250@44972",
    "361251@44973",
    "361252@44974",
    "361253@44975",
    "361254@44976",
    "361255@44977",
    "361256@44978",
    "361257@44979",
    "361258@44980",
    "361259@44981",
    "361260@44983",
    "361261@44984",
    "361264@44987",
    "361266@44989",
    "361267@44990",
    "361268@44992",
    "361269@44993",
    "361272@45012",
    "361616@41021",
    "361617@44960",
    # "361618@44962", - This dataset has string features - not supported by benchmark
    "361619@44967",
    "361621@44970",
    "361622@44994",
    "361623@45402",
]

automl_dids_regression = [
    "167210@41021",
    "233211@42225",
    "233212@42571",
    "233213@4549",
    "233214@42572",
    "233215@42570",
    "317614@42705",
    "359929@42728",
    "359930@550",
    "359931@546",
    "359932@541",
    "359933@507",
    "359934@505",
    "359935@287",
    "359936@216",
    "359937@41540",
    "359938@42688",
    "359939@422",
    "359940@416",
    "359941@42724",
    "359942@42727",
    "359943@42729",
    "359944@42726",
    "359945@42730",
    "359946@201",
    "359948@41980",
    "359949@42731",
    "359950@531",
    "359951@42563",
    "359952@574",
    "360932@3050",
    "360933@3277",
    "360945@43071",
]

# SELECTED to have no overlap with test_tabpfn and automl
# Roughly similar instance and feature distribution as automl
valid_dids_classification = [
    40707,
    42141,
    1508,
    40693,
    1483,
    1040,
    1463,
    43901,
    42140,
    981,
    184,
    459,
    45547,
    49,
    42532,
    1037,
    1128,
    41991,
    40645,
    40665,
    1060,
    143,
    1496,
    # 45575, # somehow this datasets makes our dataset loading fail (killing the kernel)
    1527,
    40677,
    1511,
    40922,
    1222,
    42585,
    679,
    4552,
    40590,
    734,
    40704,
    183,
    40910,
    40705,
    59,
    1064,
    465,
    1056,
    185,
    53,
    357,
    473,
    45548,
    45549,
    35,
    # 171,
    1465,
    41671,
    1491,
    1477,
    40588,
    51,
    42172,
    # 313, # too imbalanced for CV, multiple classes with only 1 instance
    42793,
    468,
    1459,
    45563,
    1046,
    60,
    4154,
    39,
    41083,
    40646,
    803,
    259,
    42192,
    45567,
    4153,
    1472,
    453,
    474,
    312,
    1115,
    30,
    1119,
    1479,
    1493,
    56,
    1071,
    40997,
    40669,
    40711,
    372,
    41082,
    40474,
    # 1502, # too imbalanced for CV
    340,
    375,
    41960,
    43892,
    1471,
    40593,
    40497,
    377,
    1476,
    338,
    42468,
    45578,
    40596,
    42223,
    350,
    45545,
]

# List of datasets that had to be removed because splitting into train/test with classes in both splits was not possible
valid_dids_classification_remove_due_to_splitting = [42223, 39, 474, 453]
valid_dids_classification_remove_due_to_too_large = [
    41082,  # Size: [9298, 256]
    40910,  # Size: [3686, 400]
]

valid_dids_classification = list(
    set(valid_dids_classification)
    - set(valid_dids_classification_remove_due_to_splitting)
    - set(valid_dids_classification_remove_due_to_too_large)
)

# Regression datasets are smaller in terms of features on average than classification datasets
# Not sure why, but this is the case in valid and benchmark datasets
valid_dids_regression = [
    42464,
    44150,
    41523,
    308,
    1200,
    506,
    42364,
    42559,
    41506,
    218,
    482,
    1193,
    503,
    560,
    42360,
    44793,
    1201,
    1435,
    1051,
    456,
    344,
    1575,
    40601,
    555,
    670,
    42175,
    42712,
    23516,
    42737,
    1589,
    1199,
    42183,
    23513,
    43384,
    41187,
    # 6331, contains an inf in x
    301,
    567,
    42362,
    43090,
    # 44988,
    516,
    1433,
    43466,
    43403,
    1424,
    232,
    43121,
    45536,
    43978,
    1436,
    43079,
    298,
    42636,
    42110,
    42545,
    198,
    41938,
    44052,
    1070,
    405,
    529,
    196,
    518,
    # 3040, # has nan target
    43477,
    695,
    44961,
    551,
    # 1414,  # error: BadAttributeType
    42369,
    42367,
    42366,
    45559,
    227,
    45075,
    41928,
    206,
    4532,
    42176,
    44027,
    44986,
    43963,
    41539,
    42635,
    553,
    1213,
    45064,
    512,
    43452,
    43617,
    207,
    43672,
    43078,
    41968,
    45071,
    43889,
    44152,
    210,
    42368,
    215,
    663,
    4544,
    566,
    45062,
    534,
    43927,
    43926,
    689,
    41969,
    1593,
    45074,
    1592,
    44231,
    522,
    660,
    1430,
    231,
    42821,
    543,
    43878,
    1591,
    1581,
    44028,
    511,
    1588,
    1197,
]

# List of datasets that had to be removed because splitting into train/test with classes in both splits was not possible
valid_dids_regression_remove_too_hard = [670]  # Spearman < 0.05 for lgb and ridge
# the following datasets have <= 10 unique y values
valid_dids_regression_remove_actually_classification = [
    516,
    45062,
    45075,
    298,
    301,
    1589,
    45559,
    44150,
    43384,
    42636,
    43403,
    43672,
    1433,
    1435,
    1436,
    43452,
    43466,
    43477,
    42464,
    45536,
    231,
    44793,
    506,
    287,
    44966,
    44971,
    44972,
    44055,
    44136,
]
valid_dids_regression = list(
    set(valid_dids_regression)
    - set(valid_dids_regression_remove_too_hard)
    - set(valid_dids_regression_remove_actually_classification)
)
