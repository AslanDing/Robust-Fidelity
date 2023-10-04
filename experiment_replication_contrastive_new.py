import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
print(torch.cuda.is_available())
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication import replication,replication_sp_new,replication_sp_contrastive

import nni
from nni.utils import merge_parameter

_dataset = 'ba2motifs'      # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
_explainer = 'pgexplainer' # One of: pgexplainer, gnnexplainer

# Parameters below should only be changed if you want to run any of the experiments in the supplementary
_folder = 'replication' # One of: replication, extension

# PGExplainer
config_path = f"./ExplanationEvaluation/configs/{_folder}/explainers/{_explainer}/{_dataset}.json"
print(config_path)
config = Selector(config_path)

nni_params = nni.get_next_parameter()
config.args.explainer = merge_parameter(config.args.explainer, nni_params)

extension = (_folder == 'extension')

# config.args.explainer.seeds = [0]

(auc, auc_std), inf_time = replication_sp_new(config.args.explainer, extension,
                            run_qual=False,results_store=True,extend=False)

print((auc, auc_std), inf_time)

nni.report_final_result(auc)

"""
nfid+

bashapes
pgexplainer(60)
fid_plus_mean [0.01385774 0.01535375 0.01727166 0.01971407 0.022992   0.02764082
 0.03441595 0.04591827 0.06878305 0.13189347 0.21486777]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.02098866 0.02326466 0.02617472 0.02987964 0.03485329 0.04191729
 0.05221607 0.06955185 0.1037594  0.19820829 0.32862434]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.5        0.54917458 0.59925298 0.64909949 0.69915202 0.74961962
 0.79925617 0.84926007 0.8991552  0.94918499 0.988657  ]
score: 0.9993016799871388

gnnexplainer(60)
fid_plus_mean [0.01355025 0.01500097 0.01681936 0.01901155 0.02173525 0.02568941
 0.0304753  0.03935354 0.05581636 0.10434038 0.21486777]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.02012621 0.02226035 0.02495451 0.02812729 0.03206434 0.03785645
 0.04478279 0.05748541 0.08136869 0.15013138 0.32862434]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.5        0.54917458 0.59925298 0.64909949 0.69915202 0.74961962
 0.79925617 0.84926007 0.8991552  0.94918499 0.988657  ]
score: 0.8829939770796014


bacommunity
gnnexplainer(60)
fid_plus_mean [0.00476011 0.00529813 0.00589076 0.00671107 0.00778007 0.00917664
 0.01169313 0.01563145 0.02270984 0.04119855 0.03331083]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.00366302 0.00406552 0.004549   0.00522745 0.00617185 0.00731521
 0.00928422 0.01255255 0.01769296 0.03828085 0.0467575 ]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.5        0.54940926 0.59949294 0.64938988 0.69946489 0.74965132
 0.79942704 0.84934402 0.89942245 0.94932464 0.98235972]
score: 0.7052163503055568

pgexplainer(60)
fid_plus_mean [0.00468769 0.00517446 0.00574885 0.0065353  0.00747991 0.00876001
 0.01063181 0.01349179 0.01998938 0.03523542 0.0333108 ]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.00392842 0.00446331 0.00506504 0.00577405 0.00701341 0.00841489
 0.01061808 0.01456909 0.02229062 0.04325117 0.0467575 ]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.5        0.54940926 0.59949861 0.64938988 0.69946489 0.74965132
 0.79942704 0.84934402 0.89942245 0.94932464 0.98235972]
score: 0.8640387453497266

treecycles
gnnexplainer(60)
fid_plus_mean [0.05393915 0.04973648 0.05003898 0.04698018 0.04523146 0.04560861
 0.04645007 0.05078066 0.04565512 0.05370618 0.08906554]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.11827523 0.11693482 0.11713877 0.10643652 0.10449106 0.10207191
 0.09328704 0.09156746 0.07555556 0.09722222 0.22902778]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.5        0.53925647 0.58731228 0.63486211 0.68525577 0.74183339
 0.78595399 0.8357921  0.88389749 0.93139773 0.76200367]
score: 0.6856381061106771

pgexplainer(60)
fid_plus_mean [0.07449289 0.06004096 0.06179533 0.06418525 0.06693721 0.07060324
 0.06518636 0.06782606 0.03681402 0.0283042  0.08906553]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.16459524 0.13792636 0.13490313 0.11967298 0.12348633 0.1262013
 0.1037037  0.09964286 0.03444444 0.00833333 0.22902778]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.50742188 0.5576244  0.59405097 0.65410231 0.69333014 0.7541612
 0.79422512 0.84600043 0.8974717  0.9405644  0.76200367]
score: 0.7742158865589137

treegrids
gnnexplainer(289)
fid_plus_mean [0.14803014 0.15011736 0.15295785 0.153425   0.14930186 0.14816266
 0.15079712 0.14720061 0.14551244 0.14270693 0.21453115]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.04427391 0.04635122 0.0463527  0.04402683 0.03625113 0.03523907
 0.04117922 0.03627451 0.03575548 0.03114187 0.12296507]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.5        0.54005492 0.58941101 0.63167984 0.68452913 0.74208711
 0.78411425 0.83627152 0.87923238 0.92789641 0.57261553]
score: 0.5178228309519941

pgexplainer(289)
fid_plus_mean [0.23948902 0.24860028 0.26032288 0.26314041 0.26999198 0.26120602
 0.25284716 0.24928017 0.31096184 0.38568109 0.21453115]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.17014817 0.17960235 0.18543784 0.18949037 0.19926198 0.18268523
 0.19004229 0.19256055 0.27537486 0.38927336 0.12296507]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.5042165  0.54529782 0.59467165 0.63566148 0.68621371 0.74454387
 0.78942927 0.83948765 0.88667116 0.92858845 0.57261553]
score: 0.6812721939445011

ba2motifs
pgexplainer(200)
fid_plus_mean [0.04314803 0.04245899 0.03955202 0.03918747 0.03979886 0.04020108
 0.04320163 0.04806803 0.05490348 0.05992318 0.10620512]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.07851764 0.07090991 0.0533584  0.04519522 0.035875   0.02657051
 0.02054545 0.02       0.015      0.005      0.13616026]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.51364925 0.55163576 0.60661953 0.64309645 0.70066016 0.74986572
 0.7967316  0.84444158 0.89480333 0.94259207 0.7843162 ]
score: 0.6477245007990406
pgexplainer(1000)
fid_plus_mean [0.04285188 0.04216879 0.04034724 0.03946164 0.03978638 0.03999964
 0.04199619 0.04572888 0.05357771 0.05916979 0.10520232]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.08238405 0.07516657 0.0614308  0.05254362 0.04286763 0.03390385
 0.02547273 0.02398214 0.01966667 0.00816667 0.13486923]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.51384453 0.55079835 0.60626578 0.64424487 0.7011532  0.74943188
 0.79683167 0.84469231 0.89521714 0.94303527 0.78364295]
score: 0.657355758839645


gnnexplainer(1000)
fid_plus_mean [0.06734826 0.06723325 0.06701669 0.06704696 0.06668541 0.06593254
 0.06519455 0.06341487 0.06228194 0.05797919 0.1050317 ]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.15793231 0.15572101 0.15297143 0.15083041 0.14604583 0.13961538
 0.13519091 0.12525    0.1183     0.102      0.1341    ]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.5        0.53929487 0.59782053 0.63753203 0.69605772 0.745
 0.79429489 0.84282053 0.89253203 0.9410577  0.78426412]
score: 0.6341918386500691

gnnexplainer(200)
fid_plus_mean [0.06796986 0.06743358 0.06743428 0.06683808 0.06646252 0.06737317
 0.06682672 0.06690469 0.06275738 0.05816794 0.10606067]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.15625385 0.15407609 0.15046429 0.14654971 0.14339583 0.14192308
 0.13672727 0.135      0.12       0.10666667 0.13558333]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.5        0.53928589 0.59800643 0.63750062 0.69622119 0.74485
 0.79443592 0.84285643 0.89265062 0.94107116 0.78470129]
score: 0.6248762398809076



Mutag
pgexplainer(1015)
fid_plus_mean [0.042265   0.042265   0.042265   0.04226426 0.04228687 0.04241633
 0.04262516 0.0430544  0.0469721  0.06711981 0.11447757]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.0391134  0.0391134  0.0391134  0.0391134  0.03914319 0.03921893
 0.03932482 0.03941569 0.04187136 0.05711735 0.11896552]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.88064074 0.88064074 0.88064074 0.88067082 0.88098852 0.88201341
 0.88380995 0.88852939 0.90840453 0.95035645 0.98744323]
score: 0.7660371877970388

gnnexplainer(1015)
fid_plus_mean [0.05657007 0.05794339 0.05967286 0.06110969 0.06200846 0.06344075
 0.06441114 0.06547112 0.06531984 0.06443304 0.11447757]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.04938419 0.05037292 0.05183109 0.05296058 0.05311715 0.05505984
 0.05510311 0.05559292 0.05606586 0.05631167 0.11896552]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.5        0.54048977 0.59205059 0.64120623 0.6916024  0.74517035
 0.79152136 0.84060699 0.89111149 0.94132344 0.88775234]
score: 0.5394290121851553

"""

"""
nfid-

pgexlainer
bashapes
fid_plus_mean [0.01139241 0.01099781 0.0105144  0.01001297 0.0095433  0.00907812
 0.00860992 0.00816062 0.00774311 0.00737418 0.00701316]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.01696537 0.01634609 0.01561905 0.014883   0.0142103  0.01355158
 0.01287171 0.01220759 0.01158182 0.01103112 0.01049433]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.40148137 0.3793558  0.34948378 0.3148455  0.27782378 0.23671114
 0.19250433 0.14599118 0.09824068 0.05004202 0.        ]
score: 0.9993016799871389

bacommunity
fid_plus_mean [0.00382481 0.00372033 0.00358148 0.00342986 0.00330212 0.00315883
 0.00303629 0.00291254 0.00279393 0.00268529 0.0025585 ]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.00316079 0.00298654 0.00276422 0.00265815 0.00251718 0.00237018
 0.00220861 0.00211818 0.00200768 0.00189253 0.00180067]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.34874445 0.33372974 0.31291854 0.28680375 0.25783753 0.22399962
 0.18717313 0.14380706 0.09783501 0.04980846 0.00040672]
score: 0.8640386872582763

treecycles
This model obtained: Train Acc: 0.9425, Val Acc: 0.9770, Test Acc: 0.9432.
fid_plus_mean [0.07409809 0.07409809 0.07409809 0.07409809 0.07409809 0.07409809
 0.07409809 0.07409809 0.07409809 0.07409809 0.07409809]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.1516448 0.1516448 0.1516448 0.1516448 0.1516448 0.1516448 0.1516448
 0.1516448 0.1516448 0.1516448 0.1516448]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
score: 0.7742158865589137

treegrids
fid_plus_mean [0.15892012 0.15892012 0.15892011 0.15892012 0.15892011 0.15892012
 0.15892012 0.15892012 0.15892012 0.15892011 0.15893895]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.05859228 0.05859228 0.05859228 0.05859228 0.05859228 0.05859228
 0.05859228 0.05859228 0.05859228 0.05859228 0.05865602]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.00034602 0.00034602 0.00034602 0.00034602 0.00034602 0.00034602
 0.00034602 0.00034602 0.00034602 0.00034602 0.        ]
score: 0.681272193944501

ba2motifs
fid_plus_mean [0.08842038 0.08592565 0.08360668 0.08132551 0.07777335 0.07489232
 0.07239711 0.06947427 0.06808545 0.06642564 0.05758063]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.25526461 0.24943021 0.24378852 0.23741689 0.22621455 0.21557084
 0.20734901 0.19539817 0.18761198 0.17812063 0.18295395]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.48655459 0.44836423 0.39338048 0.35680353 0.29933602 0.25013428
 0.20326842 0.1554564  0.1050005  0.05740793 0.2156838 ]
score: 0.6477559038962637

MUTAG
fid_plus_mean [-0.0041615  -0.0041615  -0.0041615  -0.00415988 -0.00414681 -0.00416562
 -0.00421567 -0.00421508 -0.0041612  -0.00336405 -0.0011494 ]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [-0.01167756 -0.01167756 -0.01167756 -0.01167756 -0.01167756 -0.01166976
 -0.0116098  -0.01154806 -0.01138833 -0.01035499 -0.00838996]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.11935926 0.11935926 0.11935926 0.11932918 0.11901148 0.11798659
 0.11619005 0.11147443 0.09159547 0.04964988 0.01255677]
score: 0.7660371877970388

gnnexplainer
bashapes
fid_plus_mean [0.0113904  0.01099709 0.01051845 0.01001441 0.00954323 0.00907608
 0.00861315 0.00815953 0.00774221 0.00737415 0.00701318]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.01696761 0.01634661 0.01562067 0.01488348 0.01420924 0.01354954
 0.01287518 0.01220537 0.01158012 0.0110304  0.01049433]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.40153766 0.37935402 0.34952056 0.3148595  0.27774387 0.23657085
 0.1926647  0.14593457 0.09819787 0.050049   0.        ]
score: 0.8829929435946903

bacommunity
fid_plus_mean [0.00381599 0.0037129  0.00357746 0.00343103 0.00329727 0.00316447
 0.00303661 0.00290916 0.00279055 0.00268486 0.00255851]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.00315995 0.00298629 0.00276399 0.00258737 0.00248072 0.00237303
 0.00223822 0.00214458 0.00200801 0.00189253 0.00180067]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.34867263 0.33370709 0.31287176 0.28686384 0.25759436 0.22406343
 0.18716638 0.14375417 0.09794621 0.04980846 0.00040672]
score: 0.7052142114839741

treecycles
fid_plus_mean [0.0740981  0.07409811 0.07409812 0.07409811 0.07409812 0.07409811
 0.07409811 0.07409811 0.07409812 0.07409811 0.07409811]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.1516448 0.1516448 0.1516448 0.1516448 0.1516448 0.1516448 0.1516448
 0.1516448 0.1516448 0.1516448 0.1516448]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
score: 0.6718184822447877

treegrids
fid_plus_mean [0.15892008 0.15892009 0.15892008 0.15892009 0.15892008 0.15892009
 0.15892008 0.15892009 0.15892008 0.15892009 0.15893892]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.05859228 0.05859228 0.05859228 0.05859228 0.05859228 0.05859228
 0.05859228 0.05859228 0.05859228 0.05859228 0.05865602]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.00034602 0.00034602 0.00034602 0.00034602 0.00034602 0.00034602
 0.00034602 0.00034602 0.00034602 0.00034602 0.        ]
score: 0.5178228309519941

ba2motifs
fid_plus_mean [0.06427027 0.06510053 0.06527412 0.06575718 0.06603303 0.06561444
 0.0660205  0.06593274 0.06664479 0.06659209 0.05758063]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.18005385 0.18060847 0.18022043 0.18069129 0.17927381 0.1767325
 0.17671341 0.17409091 0.17458937 0.17188016 0.18295395]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.5        0.46071411 0.4019936  0.36249935 0.30377884 0.25515
 0.20556411 0.1571436  0.10734935 0.05892884 0.21529871]
score: 0.6248717944333877

MUTAG
fid_plus_mean [0.01961844 0.02112261 0.02315297 0.0251778  0.02734047 0.02936935
 0.03114257 0.03279959 0.03472646 0.0363877  0.02938374]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.02114956 0.022304   0.02376571 0.02533321 0.02717828 0.02846034
 0.03004538 0.0313213  0.0326453  0.03390002 0.02550448]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.5        0.45951021 0.4079494  0.35879375 0.30839758 0.25482964
 0.20847863 0.159393   0.10888849 0.05867654 0.11224765]
score: 0.5394290088511994
"""

"""
ba2motifs GT acc 1.0

GNNExplainer
fid-
fid_plus_mean [0.02990372 0.03012479 0.03043082 0.03090299 0.03090873 0.03091383
 0.03115655 0.03144453 0.03175147 0.03246377 0.01560751]
fid_plus_label_mean [0.04336154 0.04330026 0.04427957 0.04523674 0.04465476 0.04397782
 0.04384756 0.04456169 0.04473913 0.04706904 0.        ]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.5        0.46071411 0.4019936  0.36249935 0.30377884 0.25515
 0.20556411 0.1571436  0.10734935 0.05892884 0.21529871]
score: 0.6158279873977263

PGExplainer
fid_plus_mean [0.03770575 0.03603786 0.03395709 0.03296415 0.03167679 0.03039389
 0.03040545 0.02988815 0.03159131 0.03261821 0.01560751]
fid_minus_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
fid_plus_label_mean [0.09238853 0.08585922 0.07832986 0.07307453 0.06682816 0.06135707
 0.05891939 0.05474628 0.05324193 0.05117242 0.        ]
fid_minus_label_mean [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sparsity_mean [0.48457783 0.43846807 0.38066344 0.33494741 0.26812455 0.22204981
 0.18495852 0.13329578 0.08618424 0.04640354 0.2156838 ]
score: 0.5202418323450761

"""

