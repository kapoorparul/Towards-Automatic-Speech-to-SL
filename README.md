
# Towards Automatic Speech to Sign Language Generation 

Source code for "Towards Automatic Speech to Sign Language Generation " (Parul Kapoor, Rudrabha Mukhopadhyay, Sindhu B Hegde, Vinay Namboodiri , C V Jawahar) 

Paper published at Interspeech 2021, available at https://arxiv.org/abs/2106.12790


# Usage

To run, start __main__.py with arguments "train" and ".\Configs\Base.yaml":

`python __main__.py train ./Configs/Base.yaml` 

Data preparation steps are given in the 'preprocess' folder.

To test on the test data, use:

`python __main__.py test ./Configs/Base.yaml ./Models/best.ckpt`


# Abstract

We aim to solve the highly challenging task of generating continuous sign language videos solely from speech segments for
the first time. Recent efforts in this space have focused on generating such videos from human-annotated text transcripts with-
out considering other modalities. However, replacing speech with sign language proves to be a practical solution while com-
municating with people suffering from hearing loss. Therefore, we eliminate the need of using text as input and design tech-
niques that work for more natural, continuous, freely uttered speech covering an extensive vocabulary. Since the current
datasets are inadequate for generating sign language directly from speech, we collect and release the first Indian sign lan-
guage dataset comprising speech-level annotations, text transcripts, and the corresponding sign-language videos. Next, we
propose a multi-tasking transformer network trained to generate signer’s poses from speech segments. With speech-to-text as
an auxiliary task and an additional cross-modal discriminator, our model learns to generate continuous sign pose sequences in
an end-to-end manner. Extensive experiments and comparisons with other baselines demonstrate the effectiveness of our ap-
proach. We also conduct additional ablation studies to analyze the effect of different modules of our network. A demo video
containing several results is attached to the supplementary material.

# Reference

If you wish to reference Progressive Transformers in a publication or thesis, please cite the following [paper](https://arxiv.org/abs/2106.12790):

```
@misc{kapoor2021automatic,
      title={Towards Automatic Speech to Sign Language Generation}, 
      author={Parul Kapoor and Rudrabha Mukhopadhyay and Sindhu B Hegde and Vinay Namboodiri and C V Jawahar},
      year={2021},
      eprint={2106.12790},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
<sub>This code is modified version of Progressive Transformers for End-to-End Sign Language Production code at https://github.com/BenSaunders27/ProgressiveTransformersSLP . We thank the author for this wonderful code. </sub>
