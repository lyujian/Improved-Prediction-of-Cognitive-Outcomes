# Improved Prediction of Cognitive Outcomes via Globally Aligned Imaging Biomarker Enrichments Over Progressions

## Abstract
Longitudinal brain images have been widely used to predict clinical scores for automatic diagnosis of Alzheimer’s
Disease (AD) in recent years. However, incomplete temporal neuroimaging records of the patients pose a major challenge
to use these image data for accurately diagnosing AD. 
In this paper, we propose a novel method to learn an enriched representation for imaging biomarkers, which simultaneously
captures the information conveyed by both the baseline neuroimaging records of all the participants in a studied cohort and
the progressive variations of the available follow-up records of every individual participant. 
Taking into account that different participants usually take different numbers of medical records at different time points, we develop a robust learning objective that
minimizes the summations of a number of not-squared l2 -norm distances, which, though, is difficult to efficiently solve in general.
Thus we derive a new efficient iterative algorithm with rigorously proved convergence. We have conducted extensive experiments
using the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset. 
Clear performance gains have been achieved when we predict different cognitive scores using the enriched biomarker
representations learned by our new method. We further observe that the top selected biomarkers by our proposed method are
in perfect accordance with the known knowledge in existing clinical AD studies. All these promising experimental results have
demonstrated the effectiveness of our new method.

## Code

The organization of this repository is as follows:
1. `DATA.m` preprocess the data;
2. `MIL_FS_ADNI.m` and `MIL_VBM_ADNI.m` learn the projection using VBM and FS respectively; 
3. `Performance_FS.m` and `CNN_FS_RAVLT.m` learn a enriched representation compare the enriched  representation by using different methods -- LR, RR, Lasso, SVR and CNN using modality of FS;
4. `Performance_VBM.m` and `CNN_VBM_RAVLT.m` learn a enriched representation compare the enriched  representation by using different methods -- LR, RR, Lasso, SVR and CNN using modality of VBM.
