# Vision Language Pretraining Glossary

## 1. Models

### 1.1 Encoder-Only Models
<div align=center>
    <img src="encoder_only.png">
</div>
<center>A typical encoder-only two stream VLP model</center>


* BERT-like Pretrained Family

| Model Name 	| Arxiv Time  	| Paper                                     	| Code                                                               	| Resources                                                                     	|
|------------	|-------------	|-------------------------------------------	|--------------------------------------------------------------------	|-------------------------------------------------------------------------------	|
| ViLBERT    	| Aug 6 2019  	| [paper](https://arxiv.org/abs/1908.02265) 	| [official](https://github.com/facebookresearch/vilbert-multi-task) 	|                                                                               	|
| VisualBERT 	| Aug 9 2019  	| [paper](https://arxiv.org/abs/1908.03557) 	| [official](https://github.com/uclanlp/visualbert)                  	| [huggingface](https://huggingface.co/docs/transformers/model_doc/visual_bert) 	|
| LXMERT     	| Aug 20 2019 	| [paper](https://arxiv.org/abs/1908.07490) 	| [official](https://github.com/airsplay/lxmert)                     	| [huggingface](https://huggingface.co/docs/transformers/model_doc/lxmert)      	|
| VL-BERT    	| Aug 22 2019 	| [paper](https://arxiv.org/abs/1908.08530) 	| [official](https://github.com/jackroos/VL-BERT)                    	|                                                                               	|
| UNITER     	| Sep 25 2019 	| [paper](https://arxiv.org/abs/1909.11740) 	| [official](https://github.com/ChenRocks/UNITER)                    	|                                                                               	|
| PixelBERT  	| Apr 2 2020  	| [paper](https://arxiv.org/abs/2004.00849) 	|                                                                    	|                                                                               	|
| Oscar      	| Apr 4 2020  	| [paper](https://arxiv.org/abs/2004.06165) 	| [official](https://github.com/microsoft/Oscar)                     	|                                                                               	|
| VinVL      	| Jan 2 2021  	| [paper](https://arxiv.org/abs/2101.00529) 	| [official](https://github.com/pzzhang/VinVL)                       	|                                                                               	|
| ViLT       	| Feb 5 2021  	| [paper](https://arxiv.org/abs/2102.03334) 	| [official](https://github.com/dandelin/ViLT)                       	| [huggingface](https://huggingface.co/docs/transformers/model_doc/vilt)        	|
| CLIP-ViL   	| Jul 13 2021 	| [paper](https://arxiv.org/abs/2107.06383) 	| [official](https://github.com/clip-vil/CLIP-ViL)                   	|                                                                               	|
| METER      	| Nov 3 2021  	| [paper](https://arxiv.org/abs/2111.02387) 	| [official](https://github.com/zdou0830/METER)                      	|                                                                               	|

* Contrastive Learning Family

| Model Name 	| Arxiv Time  	| Paper                                     	| Code                                                                  	| Comment                                                                          	|
|------------	|-------------	|-------------------------------------------	|-----------------------------------------------------------------------	|----------------------------------------------------------------------------------	|
| CLIP       	| Feb 26 2021 	| [paper](https://arxiv.org/abs/2103.00020) 	| [offical](https://github.com/openai/CLIP)                             	| Powerful representation learnt through large-scale image-text contrastive pairs. 	|
| ALIGN      	| Feb 11 2021 	| [paper](https://arxiv.org/abs/2102.05918) 	|                                                                       	| Impressive image-text retrieval ability.                                         	|
| FILIP      	| Nov 9 2021  	| [paper](https://arxiv.org/abs/2111.07783) 	|                                                                       	| Finer-grained representation learnt through contrastive patches.                 	|
| LiT        	| Nov 15 2021 	| [paper](https://arxiv.org/abs/2111.07991) 	| [official](https://google-research.github.io/vision_transformer/lit/) 	| Frozen image encoders proven to be effective.                                    	|
| FLIP       	| Dec 1 2022  	| [paper](https://arxiv.org/abs/2212.00794) 	| [offical](https://github.com/facebookresearch/flip)                   	| Further scaled up negative samples by masking out 95% image patches.             	|

* Large-scale Representation Learning Family

| Model Name 	| Arxiv Time  	| Paper                                     	| Code                                                             	| Comment                                                                                                                  	|
|------------	|-------------	|-------------------------------------------	|------------------------------------------------------------------	|--------------------------------------------------------------------------------------------------------------------------	|
| MDETR      	| Apr 26 2021 	| [paper](https://arxiv.org/abs/2104.12763) 	| [official](https://github.com/ashkamath/mdetr)                   	| Impressive visual grounding abilities achieved with DETR and RoBERTa                                                     	|
| ALBEF      	| Jul 16 2021 	| [paper](https://arxiv.org/abs/2107.07651) 	| [official](https://github.com/salesforce/ALBEF)                  	| BLIP's predecessor. Contrastive learning for unimodal representation followed by a multimodal transformer-based encoder. 	|
| VLMo       	| Nov 3 2021  	| [paper](https://arxiv.org/abs/2111.02358) 	| [official](https://github.com/microsoft/unilm/tree/master/vlmo)  	| Mixture of unimodal experts before multimodal experts.                                                                   	|
| BEiT-3     	| Aug 8 2022  	| [paper](https://arxiv.org/abs/2208.10442) 	| [official](https://github.com/microsoft/unilm/tree/master/beit3) 	| VLMo scaled up.                                                                                                          	|

### Encoder-Decoder Models
<div align=center>
    <img src="encoder_decoder.png">
</div>
<center>A typical encoder-decoder VLP model</center>
* Before LLM

| Model Name 	| Arxiv Time  	| Paper                                         	| Code                                                          	| Comment                                                                                            	|
|------------	|-------------	|-----------------------------------------------	|---------------------------------------------------------------	|----------------------------------------------------------------------------------------------------	|
| VL-T5      	| Feb 4 2021  	| [paper](https://arxiv.org/pdf/2102.02779.pdf) 	| [official](https://github.com/j-min/VL-T5)                    	| Unified image-text tasks with text generation, also capable of grounding.                          	|
| SimVLM     	| Aug 24 2021 	| [paper](https://arxiv.org/abs/2108.10904)     	| [official](https://github.com/YulongBonjour/SimVLM)           	| Pretrained with large-scale image-text pairs and image-text tasks with prefix LM.                  	|
| UniTab     	| Nov 23 2021 	| [paper](https://arxiv.org/abs/2111.12085)     	| [official](https://github.com/microsoft/UniTAB)               	| Unified text with bounding box outputs.                                                            	|
| BLIP       	| Jan 28 2021 	| [paper](https://arxiv.org/abs/2201.12086)     	| [official](https://github.com/salesforce/BLIP)                	| Capfilt method for bootstrapping image-text pair generation.                                       	|
| CoCa       	| May 4 2022  	| [paper](https://arxiv.org/abs/2205.01917)     	| [pytorch](https://github.com/lucidrains/CoCa-pytorch)         	| Large-scale image-text contrastive learning with text generation(LM)                               	|
| GIT        	| May 27 2022 	| [paper](https://arxiv.org/abs/2205.14100)     	| [official](https://github.com/microsoft/GenerativeImage2Text) 	| GPT-like language model conditioned on visual features extracted by pretrained ViT.                	|
| DaVinci    	| Jun 15 2022 	| [paper](https://arxiv.org/abs/2206.07699)     	| [official](https://github.com/shizhediao/DaVinci)             	| Output generation conditioned on prefix texts or prefix image. Supports text and image generation. 	|
