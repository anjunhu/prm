from typing import List, Tuple

import clip
import torch
from torch import nn

from .utils import PREDEFINED_TEMPLATES

class PromptExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._buffer_init = False
        self.with_trainable_params = False

    def init_buffer(self, clip_model):
        self._buffer_init = True

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        raise NotImplementedError()


class PredefinedPromptExtractor(PromptExtractor):
    def __init__(self, templates: List[str]):
        super().__init__()
        self.templates = templates

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        text_features_bucket = []
        for template in self.templates:
            noun_tokens = [clip.tokenize(template.format(noun)) for noun in noun_list]
            text_inputs = torch.cat(noun_tokens).to(
                clip_model.text_projection.data.device
            )
            text_features = clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_bucket.append(text_features)
        del text_inputs
        # ensemble by averaging
        text_features = torch.stack(text_features_bucket).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features


class ImageNetPromptExtractor(PredefinedPromptExtractor):
    def __init__(self):
        super().__init__(PREDEFINED_TEMPLATES["imagenet"])


class VILDPromptExtractor(PredefinedPromptExtractor):
    def __init__(self):
        super().__init__(PREDEFINED_TEMPLATES["vild"])


class LearnablePromptExtractor(PromptExtractor):
    def __init__(self, prompt_dim: int=512, prompt_shape: Tuple[int, int]=(16,16)):
        super().__init__()
        assert len(prompt_shape) == 2, "prompt_shape must be a tuple of length 2"
        # self.device = device
        self.prompt_dim = prompt_dim
        self.prompt_shape = prompt_shape
        self.prefix_prompt = self._init_prompt(self.n_prefix)
        self.suffix_prompt = self._init_prompt(self.n_suffix)
        self._buffer_init = False
        self.with_trainable_params = True

    def _init_prompt(self, length):
        if length == 0:
            return None
        prompt_tensor = torch.empty(length, self.prompt_dim)
        nn.init.normal_(prompt_tensor, std=0.02)
        return nn.Parameter(prompt_tensor)

    def init_buffer(self, clip_model, device):
        sentence = "X."
        prompt = clip.tokenize(sentence).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt) # 2,77,512
        self.register_buffer("start_signal", embedding[0, :1, :])  # 1,512
        self.register_buffer("dot_signal", embedding[0, 2:3, :])  # 1,512
        self.register_buffer("end_signal", embedding[0, 3:4, :])  # 1,512
        self.register_buffer("pad_signal", embedding[0, 4:5, :])  # 1,512
        self.noun_bucket = {}
        self._buffer_init = True

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        if not self._buffer_init:
            raise RuntimeError(
                f"Buffer of {self.__class__.__name__} is not initialized"
            )
        self._update_noun_features(noun_list, clip_model)

        prefix = [self.start_signal]
        if self.prefix_prompt is not None:
            prefix.append(self.prefix_prompt)
        prefix = torch.cat(prefix)
        suffix = [self.dot_signal, self.end_signal]
        if self.suffix_prompt is not None:
            suffix.insert(0, self.suffix_prompt)
        suffix = torch.cat(suffix)
        # only process those which are not in bucket
        # print(self.noun_bucket.keys())
        lengths = [
            len(prefix) + len(suffix) + len(self.noun_bucket[noun])
            for noun in noun_list
        ]
        embeddings = torch.stack(
            [
                torch.cat(
                    [prefix, self.noun_bucket[noun], suffix]
                    + [self.pad_signal.expand(77 - length, -1)]
                )
                for noun, length in zip(noun_list, lengths)
            ]
        )  # cls,77,512
        indices = torch.Tensor(lengths).long().to(embeddings.device) - 1
        text_features = self.get_text_feature_from_embeddings(embeddings, indices, clip_model)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features = torch.mean(text_features, dim=1)
        return text_features

    def _update_noun_features(self, noun_list, clip_model):
        left_class_names = [noun for noun in noun_list if noun not in self.noun_bucket]
        if len(left_class_names) > 0:
            with torch.no_grad():
                tokens, name_lengths = clip.tokenize(
                    left_class_names, return_length=True
                )
                name_lengths = [
                    n - 2 for n in name_lengths
                ]  # remove start end end prompt
                text_embeddings = clip_model.token_embedding(
                    tokens.to(self.device)
                ) #.type(clip_model.dtype)
                text_embeddings = [
                    embedding[1 : 1 + length]
                    for embedding, length in zip(text_embeddings, name_lengths)
                ]
            self.noun_bucket.update(
                {
                    name: embedding
                    for name, embedding in zip(left_class_names, text_embeddings)
                }
            )

    @staticmethod
    def get_text_feature_from_embeddings(x, indices, clip_model):
        cast_dtype = clip_model.transformer.get_cast_dtype()
        x = x + clip_model.positional_embedding.type(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(cast_dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), indices] @ clip_model.text_projection
        return x

    @staticmethod
    def get_text_feature_from_strings(text, clip_model):
        cast_dtype = clip_model.transformer.get_cast_dtype()
        x = clip_model.token_embedding(text).type(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + clip_model.positional_embedding.type(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(cast_dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
        return x

    @property
    def n_prefix(self):
        return self.prompt_shape[0]

    @property
    def n_suffix(self):
        return self.prompt_shape[1]

    @property
    def device(self):
        return self.start_signal.device

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """

        repr = f"prefix_prompt:{self.n_prefix},suffix_prompt:{self.n_suffix},dimension:{self.prompt_dim}\n"
        repr = repr + "[Normal_Init(mu=0,std=0.02)]"
        return repr


class LearnableTarget(PromptExtractor):
    def __init__(self, n_targets=100, prompt_dim: int=512, prompt_shape: Tuple[int, int]=(1,1)):
        super().__init__()
        assert len(prompt_shape) == 2, "prompt_shape must be a tuple of length 2"
        # self.device = device
        self.prompt_dim = prompt_dim
        self.prompt_shape = prompt_shape
        self.n_targets = n_targets
        self.prefix_prompt = self._init_prompt(self.n_prefix)
        self.suffix_prompt = self._init_prompt(self.n_suffix)
        self._buffer_init = False
        self.with_trainable_params = True

    def _init_prompt(self, length):
        if length == 0:
            return None
        prompt_tensor = torch.empty(self.n_targets, length, self.prompt_dim)
        nn.init.normal_(prompt_tensor, std=0.02)
        return nn.Parameter(prompt_tensor)

    def init_buffer(self, clip_model, device):
        sentence = "X."
        prompt = clip.tokenize(sentence).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt) # 2,77,512
        self.register_buffer("start_signal", embedding[0, :1, :])  # 1,512
        self.register_buffer("dot_signal", embedding[0, 2:3, :])  # 1,512
        self.register_buffer("end_signal", embedding[0, 3:4, :])  # 1,512
        self.register_buffer("pad_signal", embedding[0, 4:5, :])  # 1,512
        self.noun_bucket = {}
        self._buffer_init = True

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        if not self._buffer_init:
            raise RuntimeError(
                f"Buffer of {self.__class__.__name__} is not initialized"
            )
        self._update_noun_features(noun_list, clip_model)
        prefix = [self.start_signal.unsqueeze(0).expand(self.n_targets, -1, -1)]
        if self.prefix_prompt is not None:
            prefix.append(self.prefix_prompt)
        prefix = torch.cat(prefix, dim=1)
        suffix = [self.dot_signal.unsqueeze(0).expand(self.n_targets, -1, -1), 
                  self.end_signal.unsqueeze(0).expand(self.n_targets, -1, -1)]
        if self.suffix_prompt is not None:
            suffix.insert(0, self.suffix_prompt)
        suffix = torch.cat(suffix, dim=1)
        lengths = [
            prefix.shape[1] +  suffix.shape[1]  + self.noun_bucket[noun].shape[0]
            for noun in noun_list
        ]
        embeddings = []
        for n, p, s, l in zip(noun_list, prefix, suffix, lengths):
            # print(p.shape, self.noun_bucket[n].shape, s.shape)
            e = torch.cat(
                    [p, self.noun_bucket[n], s]
                    + [self.pad_signal.expand(77 - l, -1)])
            # print(e.shape)
            embeddings.append(e)
        embeddings = torch.stack(embeddings) # cls,77,512
        indices = torch.Tensor(lengths).long().to(embeddings.device) - 1
        text_features = self.get_text_feature_from_embeddings(embeddings, indices, clip_model)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def _update_noun_features(self, noun_list, clip_model):
        left_class_names = [noun for noun in noun_list if noun not in self.noun_bucket]
        if len(left_class_names) > 0:
            with torch.no_grad():
                tokens, name_lengths = clip.tokenize(
                    left_class_names, return_length=True
                )
                name_lengths = [
                    n - 2 for n in name_lengths
                ]  # remove start end end prompt
                text_embeddings = clip_model.token_embedding(
                    tokens.to(self.device)
                ) #.type(clip_model.dtype)
                text_embeddings = [
                    embedding[1 : 1 + length]
                    for embedding, length in zip(text_embeddings, name_lengths)
                ]
            self.noun_bucket.update(
                {
                    name: embedding
                    for name, embedding in zip(left_class_names, text_embeddings)
                }
            )

    @staticmethod
    def get_text_feature_from_embeddings(x, indices, clip_model):
        cast_dtype = clip_model.transformer.get_cast_dtype()
        x = x + clip_model.positional_embedding.type(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(cast_dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), indices] @ clip_model.text_projection
        return x

    @staticmethod
    def get_text_feature_from_strings(text, clip_model):
        cast_dtype = clip_model.transformer.get_cast_dtype()
        x = clip_model.token_embedding(text).type(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + clip_model.positional_embedding.type(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(cast_dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
        return x

    @property
    def n_prefix(self):
        return self.prompt_shape[0]

    @property
    def n_suffix(self):
        return self.prompt_shape[1]

    @property
    def device(self):
        return self.start_signal.device

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """

        repr = f"prefix_prompt:{self.n_prefix},suffix_prompt:{self.n_suffix},dimension:{self.prompt_dim}\n"
        repr = repr + "[Normal_Init(mu=0,std=0.02)]"
        return repr
