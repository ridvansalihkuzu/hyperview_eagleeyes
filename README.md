# PREDICTING SOIL PARAMETERS FROM HYPERSPECTRAL SATELLITE IMAGES

This project includes the soil parameter estimation algorithms based on various machine
machine learning approaches, which have been developed for the purpose of participating the [*HYPERVIEW: Seeing Beyond the Visible* Challenge](https://platform.ai4eo.eu/seeing-beyond-the-visible)

The members of the **Team Eagle Eyes** building this solution for the HYPERVIEW Challenge are:

* [Ridvan Salih Kuzu](https://scholar.google.com/citations?hl=en&user=5492gRAAAAAJ) - Helmholtz AI consultant @ German Aerospace Center (DLR)
* [Frauke Albrecht](https://www.helmholtz.ai/themenmenue/our-research/consultant-teams/helmholtz-ai-consultants-dkrz/index.html) - Helmholtz AI consultant @ German Climate Computing Centre (DKRZ)
* [Caroline Arnold](https://www.helmholtz.ai/themenmenue/our-research/consultant-teams/helmholtz-ai-consultants-dkrz/index.html) - Helmholtz AI consultant @ German Climate Computing Centre (DKRZ)
* [Roshni Kamath](https://www.helmholtz.ai/themenmenue/our-research/consultant-teams/helmholtz-ai-consultants-fzj/index.html) - Helmholtz AI consultant @ Julich Supercomputing Centre (FZJ)
* [Kai Konen](https://www.helmholtz.ai/themenmenue/our-research/consultant-teams/helmholtz-ai-consultants-dlr/index.html) - Helmholtz AI consultant @ German Aerospace Center (DLR)

[The submission file]((challenge_submission_team_eagleeyes/hyperview-main-submission_eagleeyes.ipynb)) of the **Team Eagle Eyes** has improved upon the challange baseline by %21.9,
with the first place on the [public leader-board](https://platform.ai4eo.eu/seeing-beyond-the-visible/leaderboard). 
For the further details, please refer to:

<a id="1">[1]</a> 
Albrecht, F., Arnold, C., Kamath, R.,Konen, K., Kuzu, R.S. (2022), 
[Predicting Soil Parameters from Hyperspectral Satellite Images](challenge_submission_team_eagleeyes/hyperview_for_ICIP_camera_ready_eagleeyes.pdf), 
in 29th IEEE International Conference on Image Processing (IEEE ICIP 2022), Bordeaux, France.

## FOLDER STRUCTURE

* The starter pack notebook provided by the Challenge Organization Committee is given under folder [challenge_official_starter_pack](challenge_official_starter_pack).
* The final submission of the **Team Eagle Eyes** is given under folder [challenge_submission_team_eagleeyes](challenge_submission_team_eagleeyes).
* The Vision Transformer (ViT-L/14) based soil parameter estimation experiments are given under folder [experimental_1](experimental_1).
* The Swin Transformer (Swin-T) and other DNN based soil parameter estimation experiments are given under folder [experimental_2](experimental_2).
* The Random Forest and other classical machine learning based soil parameter estimation experiments are given under folder [experimental_3](experimental_3).


## INSTALLATION AND RUNNING

If you want to run the project on docker containers, pull the customly built images for this project:
```bash
$ docker pull ridvansalih/clip:latest
$ docker pull ridvansalih/hyperview:latest
```

If you want to run the project on singularity containers, convert the docker images to singularity images as described below:
```bash
$ export SINGULARITY_CACHEDIR=$(mktemp -d -p ${PWD})
$ export SINGULARITY_TMPDIR=$(mktemp -d -p ${PWD})
$ singularity pull clip_latest.sif docker://ridvansalih/clip:latest
$ singularity pull hyperview_latest.sif docker://hyperview:latest
```

After having the docker images, you can refer to the `bash` scripts (e.g. `script_run_docker.sh`, `script_run_singular.sh`) under the experiment folders
in order to run them.

**NOTE:** Please be sure that you download the data and update the training and test folder paths in the `bash` scripts. 

**NOTE:** Please be sure that the arguments given to `main_hyper_view_training.py` scripts in each experiment folder are valid for your hardware limitations. 

## THE APPROACH

### Abstract
The <span class="smallcaps">AI4EO Hyperview</span> challenge seeks
machine learning methods that predict agriculturally relevant soil
parameters (K, Mg, P<sub>2</sub>O<sub>5</sub>, pH) from airborne
hyperspectral images. We present a hybrid model fusing Random Forest and
K-nearest neighbor regressors that exploit the average spectral
reflectance, as well as derived features such as gradients, wavelet
coefficients, and Fourier transforms. The solution is computationally
lightweight and improves upon the challenge baseline by 21.9%, with the
first place on the public leaderboard. In addition, we discuss neural
network architectures and potential future improvements.

**Index Terms—** hyperspectral images, random forests, ar-
tificial neural networks, soil parameter estimation, regression.

## 1. Introduction

Machine learning methods are employed widely in remote sensing . In
particular, agricultural monitoring via remote sensing draws significant
attention for various purposes ranging from early forecasting of crop
yield amount to the estimation of soil composite .

Predicting the fertility indicators of soil, such as percentage of
organic matter, or amount of fertilizer, is one of the leading research
topics in earth observation due to the emerging needs for improving the
agricultural efficiency without harming nature. Particularly, the
European Union Green Deal gives special importance to supporting
conventional farming practices with earth observation (EO) and
artificial intelligence (AI) for resilient production as well as healthy
soil and biodiversity .

The AI4EO platform seeks to bridge the gap between the AI and EO
communities . In the <span class="smallcaps">AI4EO Hyperview</span>
challenge, the objective is to predict soil properties from
hyperspectral satellite images, including potassium (K), magnesium (Mg),
and phosphorus pentoxide (P<sub>2</sub>O<sub>5</sub>) content, and the
pH value . The winning solution of the challenge will be running
on-board the Intuition-1 satellite.

In this manuscript, we present the solution to the <span class="smallcaps">AI4EO Hyperview</span> challenge developed by Team
<span class="smallcaps">EagleEyes</span>. Section [2](#sec_data_set) discusses the
hyperspectral image dataset, Section [3](#sec_methods) covers feature
engineering and experimental protocols for different learning
strategies, and Section [4](#sec_results) presents the preliminary
performance results for predicting the given four soil properties.
Eventually, we conclude in Section [5](#sec_discussion) and give an
outlook on future work.

## <a id="sec_data_set" /> 2. Dataset  

The hyperspectral images are taken from airborne measurements from an
unspecified location in Poland. In total, 1732 patches are available for
training, and 1154 patches remain for testing. Each patch contains
150 hyperspectral bands, spanning 462 − 492 nm with a spectral
resolution of 3.2 nm .

Samples in the dataset have been segmented into patches according to the
boundaries of the agricultural fields. As shown in Figure
[1](#FIG_field_distribution), the patch size distribution is skewed:
About one third of the samples is composed of 11 × 11 px patches, and
60% of the patches are less than 50 px wide.






<div class="center">
<figure>
<img src="/challenge_submission_team_eagleeyes/feature_examples/field_distribution.png" id="FIG_field_distribution"
alt="Distribution of dataset in terms of different patch sizes." />
</figure>
</div>
<center><strong style="color: orange; opacity: 0.80;">Figure 1: Distribution of dataset in terms of different patch sizes.</strong></center>

&nbsp;
<br />

The training data provides ground truth for all four soil parameters.
The target values for P<sub>2</sub>O<sub>5</sub> lie in the range of
[20.3−325], for K in [21.1−625], for Mg in [26.8−400], and for pH
in [5.6−7.8]. As shown in Figure [2](#FIG_target_distribution), for P<sub>2</sub>O<sub>5</sub>
and K, the target values follow a log-normal distribution with positive
skewness, while the Mg and pH values are more Gaussian distributed.
Besides, pH measurements are mostly clustered in the intervals of 0.1.

<div class="center">
<figure>
<img src="/challenge_submission_team_eagleeyes/feature_examples/target_distribution.png" id="FIG_target_distribution"
alt="Distribution of target values for each soil parameter." />
</figure>
</div>
<center><strong style="color: orange; opacity: 0.80;">Figure 2: Distribution of target values for each
soil parameter.</strong></center>
&nbsp;
<br />


## <a id="sec_methods" /> 3. Experimental Framework 

In this section, we present the feature engineering approaches,
experimental protocols for different learning strategies and metrics
utilized for validating our approach.

### Data Processing and Augmentation

#### Feature engineering for traditional ML approaches

As mentioned in Section [2](#sec_data_set), the samples are
3-dimensional patches with dimension (*w* × *h* × *c*) where width
(*w*), and height (*h*) have varying sizes, but channel (*c*) is fixed
to represent 150 spectral bands.


<div class="center">
<figure>
<p align="center">
<img src="/challenge_submission_team_eagleeyes/feature_examples/eaglepaper_reflectance.png"
id="FIG_average_reflectance"
alt="Comparison of average reflectance for different agricultural fields for each soil parameter." />
</p>
</figure>
</div>

<center><strong style="color: orange; opacity: 0.80;">
Figure 3: Comparison of average reflectance for
different agricultural fields for each soil parameter.</strong></center>

&nbsp;
<br />

Figure [3](#FIG_average_reflectance) shows the average reflectance as a function of wavelength
for the samples with minimum and maximum value of the respective soil
parameter. These average reflectance curves are remarkably dissimilar
for different values of the target variable. Thus, we use the average
reflectance as a base feature in our experiments, and derive additional
features from it. The list of the features is as follows:

-   *average reflectance*, its 1<sup>*st*</sup>, 2<sup>*nd*</sup>
    and 3<sup>*rd*</sup> order derivatives, (\[1×150\] dimension for
    each, \[1×600\] in total),

-   discrete wavelet transforms of *average reflectance* with Meyer
    wavelet : 1<sup>*s**t*</sup>, 2<sup>*nd*</sup>,
    3<sup>*rd*</sup>, 4<sup>*th*</sup> level *approximation* and
    *detail* coefficients (\[1×300\] dims. in total),
    


-   for each channel (*c*) of a field patch (*P*), singular value decomposition (SVD) 
has been conducted: $P\_{(w \\times h)} = U \\Sigma V^{T}$, in which *Σ* is square diagonal of size \[*r*×*r*\] where *r* ≤ *m**i**n*{*w*, *h*}. The first 5
diagonal values (*σ*<sub>1</sub>, *σ*<sub>2</sub>, *σ*<sub>3</sub>,
*σ*<sub>4</sub>, *σ*<sub>5</sub> ∈ *Σ*) from each channel are selected
as features (\[1×750\] dims. in total),

-   the ratio of 1<sup>*s**t*</sup>, 2<sup>*n**d*</sup> diagonals:
    *σ*<sub>1</sub>/*σ*<sub>2</sub> (\[1×150\] dims.),

-   Fast Fourier transform (FFT) of *average reflectance* and FFT of
    *σ*<sub>1</sub>/*σ*<sub>2</sub>: real and imaginary parts are
    included (\[1×600\] dims. in total).


To sum up, for each field patch, a \[1×2400\] dimensional feature array
is extracted. Some of those features for different agricultural fields
are illustrated in Figure [\[fig:feature-engineering\]][1]. For data
augmentation, 1% random Gaussian noise is added to both input features
and target values.

<div class="figure*">

<img src="figures/1st_derivative.png" alt="image" />

<img src="figures/wavelet_approximation.png" alt="image" />

<img src="figures/s1.png" alt="image" />

<img src="figures/fft_s0_real.png" alt="image" />

</div>

#### Feature engineering for deep learning approaches

For experimenting on neural networks, either a raw patch,
*P*<sub>(*w*×*h*×*c*)</sub>, or random patch subsets or pixel subsets
from the raw patch is treated as a feature from a field. For data
augmentation, we randomly add Gaussian noise, scale, crop, and rotate
the field patches.

For both feature engineering approaches, we experimented with different
data normalization techniques, including min-max scaling, standard
scaling, and robust scaling.


### Exploited Models

During development, classical machine learning approaches, such as
Random Forest (RF), K-Nearest Neighbour (KNN) and eXtreme Gradient
Boosting (XGBoost) regressors were investigated. Additionally, different
neural network architectures were explored. Since the final solution is
supposed to run on the Intuition-1 satellite, solutions that require low
computational resources are of special interest.

#### Classical machine learning architectures

We used the <span class="smallcaps">RandomForestRegressor</span> (RF)
and <span class="smallcaps">KNeighborsRegressor</span> (KNN) implemented
in the scikit-learn package , as well as <span
class="smallcaps">XGBoost</span> regressors. Since the latter does not
support multiple-regression problems, the <span
class="smallcaps">MultiOutputRegressor</span>, also from the
scikit-learn package, was wrapped around the <span
class="smallcaps">XGBoost</span>. For all model types, hyperparameter
tuning was conducted using <span class="smallcaps">Optuna</span> with
Bayesian optimization. However, we found the default parameters
performed best and only changed the number of estimators to 1000. For
all of our experiments, RFs perform better than the XGBoost and KNN
algorithms.

#### Deep Neural networks

We experimented with various neural network architectures, including
Transformers , MobileNets , CapsuleNets , multilayer perceptrons, as
well as autoencoder architectures and attention networks, such as
PSE+LTAE . To exploit the pretrained weights of the networks, we
experimented with several input modalities:

-   channel-wise dimensional reduction from (*w* × *h* × 150) to
    (*w* × *h* × 3) via convolution operation and later feeding the
    samples into the pre-trained models;

-   dropping the input layers of the pretrained networks mentioned
    above, and attaching our custom input layers that accept
    (*w* × *h* × 150) dimensional samples;

-   feeding each channel (*w* × *h* × 1) of a sample into the 150
    parallel and weight-sharing pretrained networks;

-   expanding the input dimension as (*w* × *h* × 150 × 1) to exploit 3D
    neural networks such as CapsuleNets ;

-   flattening the input dimensions, or subsampling the input for
    exploiting 1D neural networks such as multilayer perceptrons or
    autoencoders.

Nonetheless, many of those trials performed worse than the RF, except
for the Transformers with the pretrained weights (ImageNet-21k for
Swin-T and CLIP for ViT-L/14 ), when the channel-wise dimensional
reduction operation was attached to it as an input layer.

For designing those experiments, we used the Keras framework with
Tensorflow version 2.8.0 and the Pytorch framework version 1.10.0 .

### Evaluation Metrics

The evaluation metric takes into account the improvement upon the
baseline of predicting the average of each soil parameter
(MSE<sub>bl</sub>). For a given algorithm, it is calculated as:

$$\\mathrm{Score} = \\frac 1 4
                   \\sum\\limits\_{i=1}^{4}
                   \\frac{\\mathrm{MSE\_{algo}}^{(i)}}{\\mathrm{MSE\_{bl}}^{(i)}}, \\,\\, \\textrm{ where:}$$

$$\\mathrm{MSE\_{algo}}^{(i)}
    =
    \\frac 1 N \\sum \\limits\_{j=1}^N
    (p_j^{(i)} - \\hat p_j ^{(i)})^2
    .$$

## <a id="sec_results" /> 4. Results and Discussion 

Among our experiments listed in Section
[\[SUBSEC:exploited-models\]][1], the best performing ones on the public
leaderboard of the challenge are:

-   RF regression, by achieving 0.79476

-   Swin Transformer (Swin-T), by achieving 0.80028

-   Vision Transformer (ViT-L/14), by achieving 0.78799

The models show comparable performance, however, the RF is
computationally more lightweight. Since this would be advantageous for
running the model on the target Intuition-1 satelite, we selected the RF
for further optimization.

As summarized in Table [\[TAB:random-forest-scores\]][2], the average of
5-fold cross validation on the training set with RF yields a validation
score of 0.811. Note that while we improve on the baseline for all four
soil parameters, the performance varies. Mg is predicted best (0.734),
and P<sub>2</sub>O<sub>5</sub> is predicted worst (0.874).




<div class="adjustbox">

width=1

<div id="TAB:random-forest-scores">

|            **Field Edge (pixel)**            | **\# of Fields** |  **P2O5**   |   **K**   |  **Mg**   |  **pH**   | **Average** |
|:--------------------------------------------:|:----------------:|:-----------:|:---------:|:---------:|:---------:|:-----------:|
|                   **0-11**                   |       650        |    1.050    |   1.008   |   1.019   |   0.866   |    0.985    |
|                  **11-40**                   |        94        |    0.491    |   0.581   |   0.539   |   0.981   |    0.648    |
|                  **40-50**                   |       326        |    0.724    |   0.754   |   0.416   |   0.777   |    0.668    |
|                  **50-100**                  |       138        |    0.683    |   0.660   |   0.618   |   0.749   |    0.677    |
|                 **100-110**                  |       113        |    0.911    |   0.591   |   0.398   |   0.764   |    0.665    |
|                 **110-120**                  |       118        |    0.883    |   0.812   |   0.614   |   0.731   |    0.760    |
|                 **120-130**                  |       132        |    0.895    |   0.776   |   0.644   |   0.656   |    0.742    |
|                   **130+**                   |       161        |    0.808    |   0.761   |   0.842   |   0.790   |    0.801    |
|              **Entire Fields**               |     **1732**     |  **0.874**  | **0.828** | **0.734** | **0.807** |  **0.811**  |

Cross validation with RF (the lower score is better).

</div>

</div>

For the RF regression, the feature importance can be determined.
Figure [\[FIG:feature-weights\]][1] shows the derivatives of the average
spectral reflectance contribute the most, followed by the features
derived from SVD and FFT. Figure [\[FIG:channel-weights\]][2] shows the
importance of the spectral bands. The bands in the 650 − 670 nm, and
those exceeding 850 nm are considered to be most important.


<div class="center">

<figure>
<img src="figures/channel_weights.png" id="FIG:channel-weights"
alt="Hyperspectral band importances for RF regressor." />
<figcaption aria-hidden="true">Hyperspectral band importances for RF
regressor.</figcaption>
</figure>

</div>

In order to analyze if the data skewness affects performance, the
prediction scores are reported for different patch sizes in Table
[\[TAB:random-forest-scores\]][1]. Thus, we observe that the smaller
patches ( ≤ 11 × 11 px) are the major source of the prediction error.
This might stem from the higher variations in channel-aggregation due to
a lower number of pixels. For mitigating this error source, alternative
hyperparameter spaces and ML architectures were sought for smaller field
patches. KNN regression with *k* ≥ 35 improved performance on the
smaller patches (from 0.985 to 0.915) but, on the other hand, it
performs worse than RF on larger patches.

<div class="adjustbox">

width=1

<div id="TAB:hybrid-scores">

|            **Field Edge (pixel)**            | **Model**  |  **P2O5**   |   **K**   |  **Mg**   |  **pH**   | **Average** |
|:--------------------------------------------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:-----------:|
|                   **0-11**                   |    KNN     |    1.002    |   0.953   |   0.993   |   0.710   |    0.915    |
|                   **11+**                    |     RF     |    0.766    |   0.720   |   0.564   |   0.772   |    0.706    |
|              **Entire Fields**               | **Hybrid** |  **0.855**  | **0.807** | **0.725** | **0.749** |  **0.793**  |

Cross validation with hybrid regressor (RF + KNN).

</div>

</div>


Therefore, a hybrid soil paramater estimator is proposed, combining KNN
and RF regressors. Table [\[TAB:hybrid-scores\]][1] summarizes the
performance of the hybrid model in which KNN predicts the soil
parameters for smaller fields (mean edge length  ≤ 11 px), while RF
makes predictions for larger fields (mean edge length  \> 11 px). Thus,
cross-validation performance on training set has been improved from
0.811 to 0.793. With this hybrid model, our team has preserved the top
position in the public leaderboard by outperforming our former RF
regressor (from 0.79476 to 0.78113).

Figure [1][] shows density plots of the true vs the predicted soil
parameters for the validation set. For all target variables, average
values are predicted close to the 1 : 1 line, while extreme values are
hard to estimate correctly.

<div class="center">

<figure>
<img src="figures/out_prediction.png" id="FIG:pred-vs-target"
alt="Ground-truths vs predicted soil parameters." />
<figcaption aria-hidden="true">Ground-truths vs predicted soil
parameters.</figcaption>
</figure>

</div>

## 5. <a id="sec_discussion" /> Conclusion and Future Work 

In this paper, we demonstrated our solution to the <span
class="smallcaps">AI4EO Hyperview</span> challenge which seeks for the
most efficient approach to predict soil parameters (K, Mg,
P<sub>2</sub>O<sub>5</sub>, pH). With comprehensive feature engineering,
and by building a hybrid solution based on the fusion of KNN and RF
regression models, we achieved 21.9% improvement compared to the
baseline, and preserved the leadership so far. In the future, we will
select features and train models individually for the four soil
parameters to optimize performance. Besides, we will conduct further
experiments with novel architectures tailored to the provided challenge
dataset.

## Acknowledgments

We thank Lichao Mou for helpful discussions. This work was supported by
the Helmholtz Association’s Initiative and Networking Fund through
Helmholtz AI \[grant number: ZT-I-PF-5-01\] and on the HAICORE@FZJ
partition.



