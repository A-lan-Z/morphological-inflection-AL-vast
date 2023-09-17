"""
train
"""
import os
from functools import partial

import torch
from tqdm import tqdm

import dataloader
import model
import transformer
import util
from decoding import Decode, get_decode_fn
from trainer import BaseTrainer

import random

tqdm.monitor_interval = 0

tqdm = partial(tqdm, bar_format="{l_bar}{r_bar}")


class Data(util.NamedEnum):
    g2p = "g2p"
    p2g = "p2g"
    news15 = "news15"
    histnorm = "histnorm"
    sigmorphon16task1 = "sigmorphon16task1"
    sigmorphon17task1 = "sigmorphon17task1"
    sigmorphon19task1 = "sigmorphon19task1"
    sigmorphon19task2 = "sigmorphon19task2"
    lemma = "lemma"
    lemmanotag = "lemmanotag"
    lematus = "lematus"
    unimorph = "unimorph"


class Arch(util.NamedEnum):
    soft = "soft"  # soft attention without input-feeding
    hard = "hard"  # hard attention with dynamic programming without input-feeding
    approxihard = "approxihard"  # hard attention with REINFORCE approximation without input-feeding
    softinputfeed = "softinputfeed"  # soft attention with input-feeding
    largesoftinputfeed = (
        "largesoftinputfeed"  # soft attention with uncontrolled input-feeding
    )
    approxihardinputfeed = "approxihardinputfeed"  # hard attention with REINFORCE approximation with input-feeding
    hardmono = "hardmono"  # hard monotonic attention
    hmm = "hmm"  # 0th-order hard attention without input-feeding
    hmmfull = "hmmfull"  # 1st-order hard attention without input-feeding
    transformer = "transformer"
    universaltransformer = "universaltransformer"
    tagtransformer = "tagtransformer"
    taguniversaltransformer = "taguniversaltransformer"


class Trainer(BaseTrainer):
    """docstring for Trainer."""

    def set_args(self):
        """
        get_args
        """
        # fmt: off
        super().set_args()
        parser = self.parser
        parser.add_argument('--dataset', required=True, type=Data, choices=list(Data))
        parser.add_argument('--max_seq_len', default=128, type=int)
        parser.add_argument('--max_decode_len', default=128, type=int)
        parser.add_argument('--decode_beam_size', default=5, type=int)
        parser.add_argument('--init', default='', help='control initialization')
        parser.add_argument('--dropout', default=0.2, type=float, help='dropout prob')
        parser.add_argument('--embed_dim', default=100, type=int, help='embedding dimension')
        parser.add_argument('--nb_heads', default=4, type=int, help='number of attention head')
        parser.add_argument('--src_layer', default=1, type=int, help='source encoder number of layers')
        parser.add_argument('--trg_layer', default=1, type=int, help='target decoder number of layers')
        parser.add_argument('--src_hs', default=200, type=int, help='source encoder hidden dimension')
        parser.add_argument('--trg_hs', default=200, type=int, help='target decoder hidden dimension')
        parser.add_argument('--label_smooth', default=0., type=float, help='label smoothing coeff')
        parser.add_argument('--tie_trg_embed', default=False, action='store_true', help='tie decoder input & output embeddings')
        parser.add_argument('--arch', required=True, type=Arch, choices=list(Arch))
        parser.add_argument('--nb_sample', default=2, type=int, help='number of sample in REINFORCE approximation')
        parser.add_argument('--wid_siz', default=11, type=int, help='maximum transition in 1st-order hard attention')
        parser.add_argument('--indtag', default=False, action='store_true', help='separate tag from source string')
        parser.add_argument('--decode', default=Decode.greedy, type=Decode, choices=list(Decode))
        parser.add_argument('--mono', default=False, action='store_true', help='enforce monotonicity')
        parser.add_argument('--bestacc', default=False, action='store_true', help='select model by accuracy only')
        # fmt: on

    def load_data(self, dataset, train, dev, test):
        assert self.data is None
        logger = self.logger
        params = self.params
        # fmt: off
        if params.arch == Arch.hardmono:
            if dataset == Data.sigmorphon17task1:
                self.data = dataloader.AlignSIGMORPHON2017Task1(train, dev, test, params.shuffle)
            elif dataset == Data.g2p:
                self.data = dataloader.AlignStandardG2P(train, dev, test, params.shuffle)
            elif dataset == Data.news15:
                self.data = dataloader.AlignTransliteration(train, dev, test, params.shuffle)
            else:
                raise ValueError
        else:
            if dataset == Data.sigmorphon17task1:
                if params.indtag:
                    self.data = dataloader.TagSIGMORPHON2017Task1(train, dev, test, params.shuffle)
                else:
                    self.data = dataloader.SIGMORPHON2017Task1(train, dev, test, params.shuffle)
            elif dataset == Data.unimorph:
                if params.indtag:
                    self.data = dataloader.TagUnimorph(train, dev, test, params.shuffle)
                else:
                    self.data = dataloader.Unimorph(train, dev, test, params.shuffle)
            elif dataset == Data.sigmorphon19task1:
                assert isinstance(train, list) and len(train) == 2 and params.indtag
                self.data = dataloader.TagSIGMORPHON2019Task1(train, dev, test, params.shuffle)
            elif dataset == Data.sigmorphon19task2:
                assert params.indtag
                self.data = dataloader.TagSIGMORPHON2019Task2(train, dev, test, params.shuffle)
            elif dataset == Data.g2p:
                self.data = dataloader.StandardG2P(train, dev, test, params.shuffle)
            elif dataset == Data.p2g:
                self.data = dataloader.StandardP2G(train, dev, test, params.shuffle)
            elif dataset == Data.news15:
                self.data = dataloader.Transliteration(train, dev, test, params.shuffle)
            elif dataset == Data.histnorm:
                self.data = dataloader.Histnorm(train, dev, test, params.shuffle)
            elif dataset == Data.sigmorphon16task1:
                if params.indtag:
                    self.data = dataloader.TagSIGMORPHON2016Task1(train, dev, test, params.shuffle)
                else:
                    self.data = dataloader.SIGMORPHON2016Task1(train, dev, test, params.shuffle)
            elif dataset == Data.lemma:
                if params.indtag:
                    self.data = dataloader.TagLemmatization(train, dev, test, params.shuffle)
                else:
                    self.data = dataloader.Lemmatization(train, dev, test, params.shuffle)
            elif dataset == Data.lemmanotag:
                self.data = dataloader.LemmatizationNotag(train, dev, test, params.shuffle)
            else:
                raise ValueError
        # fmt: on
        logger.info("src vocab size %d", self.data.source_vocab_size)
        logger.info("trg vocab size %d", self.data.target_vocab_size)
        logger.info("src vocab %r", self.data.source[:500])
        logger.info("trg vocab %r", self.data.target[:500])

    def build_model(self):
        assert self.model is None
        params = self.params
        if params.arch == Arch.hardmono:
            params.indtag, params.mono = True, True
        kwargs = dict()
        kwargs["src_vocab_size"] = self.data.source_vocab_size
        kwargs["trg_vocab_size"] = self.data.target_vocab_size
        kwargs["embed_dim"] = params.embed_dim
        kwargs["nb_heads"] = params.nb_heads
        kwargs["dropout_p"] = params.dropout
        kwargs["tie_trg_embed"] = params.tie_trg_embed
        kwargs["src_hid_size"] = params.src_hs
        kwargs["trg_hid_size"] = params.trg_hs
        kwargs["src_nb_layers"] = params.src_layer
        kwargs["trg_nb_layers"] = params.trg_layer
        kwargs["nb_attr"] = self.data.nb_attr
        kwargs["nb_sample"] = params.nb_sample
        kwargs["wid_siz"] = params.wid_siz
        kwargs["label_smooth"] = params.label_smooth
        kwargs["src_c2i"] = self.data.source_c2i
        kwargs["trg_c2i"] = self.data.target_c2i
        kwargs["attr_c2i"] = self.data.attr_c2i
        model_class = None
        indtag, mono = True, True
        # fmt: off
        fancy_classfactory = {
            (Arch.hardmono, indtag, mono): model.HardMonoTransducer,
            (Arch.soft, indtag, not mono): model.TagTransducer,
            (Arch.hard, indtag, not mono): model.TagHardAttnTransducer,
            (Arch.hmm, indtag, not mono): model.TagHMMTransducer,
            (Arch.hmm, indtag, mono): model.MonoTagHMMTransducer,
            (Arch.hmmfull, indtag, not mono): model.TagFullHMMTransducer,
            (Arch.hmmfull, indtag, mono): model.MonoTagFullHMMTransducer,
        }
        regular_classfactory = {
            Arch.soft: model.Transducer,
            Arch.hard: model.HardAttnTransducer,
            Arch.softinputfeed: model.InputFeedTransducer,
            Arch.largesoftinputfeed: model.LargeInputFeedTransducer,
            Arch.approxihard: model.ApproxiHardTransducer,
            Arch.approxihardinputfeed: model.ApproxiHardInputFeedTransducer,
            Arch.hmm: model.HMMTransducer,
            Arch.hmmfull: model.FullHMMTransducer,
            Arch.transformer: transformer.Transformer,
            Arch.universaltransformer: transformer.UniversalTransformer,
            Arch.tagtransformer: transformer.TagTransformer,
            Arch.taguniversaltransformer: transformer.TagUniversalTransformer,
        }
        # fmt: on
        if params.indtag or params.mono:
            model_class = fancy_classfactory[(params.arch, params.indtag, params.mono)]
        else:
            model_class = regular_classfactory[params.arch]
        self.model = model_class(**kwargs)
        if params.indtag:
            self.logger.info("number of attribute %d", self.model.nb_attr)
            self.logger.info("dec 1st rnn %r", self.model.dec_rnn.layers[0])
        if params.arch in [
            Arch.softinputfeed,
            Arch.approxihardinputfeed,
            Arch.largesoftinputfeed,
        ]:
            self.logger.info("merge_input with %r", self.model.merge_input)
        self.logger.info("model: %r", self.model)
        self.logger.info("number of parameter %d", self.model.count_nb_params())
        self.model = self.model.to(self.device)

    def dump_state_dict(self, filepath):
        util.maybe_mkdir(filepath)
        self.model = self.model.to("cpu")
        torch.save(self.model.state_dict(), filepath)
        self.model = self.model.to(self.device)
        self.logger.info(f"dump to {filepath}")

    def load_state_dict(self, filepath):
        state_dict = torch.load(filepath)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.logger.info(f"load from {filepath}")

    def setup_evalutator(self):
        arch, dataset = self.params.arch, self.params.dataset
        if arch == Arch.hardmono:
            if dataset == Data.news15:
                self.evaluator = util.PairTranslitEvaluator()
            elif dataset == Data.sigmorphon17task1:
                self.evaluator = util.PairBasicEvaluator()
            elif dataset == Data.g2p:
                self.evaluator = util.PairG2PEvaluator()
            else:
                raise ValueError
        else:
            if dataset == Data.news15:
                self.evaluator = util.TranslitEvaluator()
            elif dataset == Data.g2p:
                self.evaluator = util.G2PEvaluator()
            elif dataset == Data.p2g:
                self.evaluator = util.P2GEvaluator()
            elif dataset == Data.histnorm:
                self.evaluator = util.HistnormEvaluator()
            else:
                self.evaluator = util.BasicEvaluator()

    def evaluate(self, mode, batch_size, epoch_idx, decode_fn):
        self.model.eval()
        sampler, nb_batch = self.iterate_batch(mode, batch_size)
        results = self.evaluator.evaluate_all(
            sampler, batch_size, nb_batch, self.model, decode_fn
        )
        for result in results:
            self.logger.info(
                f"{mode} {result.long_desc} is {result.res} at epoch {epoch_idx}"
            )
        return results

    def entropy(self, predictions):
        # Filter out probabilities less than or equal to 0.05
        predictions = predictions[predictions > 0.05]

        # Calculate entropy
        ent = -torch.sum(predictions * torch.log(predictions))
        return ent.item()

    def decode(self, mode, batch_size, write_fp, decode_fn):
        self.model.eval()
        cnt = 0
        sampler, nb_batch = self.iterate_batch(mode, batch_size)
        nll_list = []
        entropy_list = []
        with open(f"{write_fp}.{mode}.tsv", "w") as fp:
            if self.params.decode == Decode.ensemble:
                fp.write("target\tall_predictions\tall_log_probs\tdenormalized_probs\n")
            else:
                fp.write("prediction\ttarget\tloss\tdist\tnll\tentropy\n")

            for src, src_mask, trg, trg_mask in tqdm(
                    sampler(batch_size), total=nb_batch
            ):
                pred, nlls, entropies, _ = decode_fn(self.model, src, src_mask)
                if self.params.decode == Decode.ensemble:
                    trg = util.unpack_batch(trg)
                    for t, all_sequences, all_log_probs, probs in zip(trg, pred, nlls, entropies):
                        t = self.data.decode_target(t)

                        # Preprocess each prediction in all_sequences
                        all_sequences = util.unpack_batch(all_sequences)
                        all_sequences = [" ".join(self.data.decode_target(seq)) for seq in all_sequences]

                        # Convert each tensor in all_log_probs to a regular Python value
                        all_log_probs = [str(log_prob.cpu().item()) for log_prob in all_log_probs]
                        probs = [str(prob.cpu().item()) for prob in probs]

                        fp.write(f'{" ".join(t)}\t{"|".join(all_sequences)}\t{"|".join(all_log_probs)}\t{"|".join(probs)}\n')
                        cnt += 1
                else:
                    self.evaluator.add(src, pred, trg)
                    nll_list.extend(nlls)
                    entropy_list.extend(entropies)

                    data = (src, src_mask, trg, trg_mask)
                    losses = self.model.get_loss(data, reduction=False).cpu()

                    pred = util.unpack_batch(pred)
                    trg = util.unpack_batch(trg)
                    for p, t, loss, nll, entropy in zip(pred, trg, losses, nlls, entropies):
                        dist = util.edit_distance(p, t)
                        p = self.data.decode_target(p)
                        t = self.data.decode_target(t)
                        fp.write(f'{" ".join(p)}\t{" ".join(t)}\t{loss.item()}\t{dist}\t{nll}\t{entropy}\n')
                        cnt += 1

        self.logger.info(f"finished decoding {cnt} {mode} instance")
        if self.params.decode != Decode.ensemble:
            results = self.evaluator.compute(reset=True)
            return results, nll_list, entropy_list
        else:
            return None, None, None

    def al_sampling(self, criteria, criteria_list, num_samples=100, beta=1.0):
        print(f"Starting al_sampling with criteria: {criteria}, num_samples: {num_samples}, beta: {beta}")

        # If the criteria is "information_density", read the kor_density.tsv and compute the information density
        if criteria == "information_density":
            with open('kor_density.tsv', 'r', encoding="utf-8") as f:
                lines = f.readlines()
                # Extracting density values from the file
                density_values = [float(line.strip().split('\t')[1]) for line in lines]
                print(f"Read {len(density_values)} density values from kor_density.tsv")
                # Multiplying density with entropy to get information density
                criteria_list = [entropy * (density ** (-beta)) for density, entropy in
                                 zip(density_values, criteria_list)]

        # Before sorting and selecting uncertain samples
        print(f"Size of criteria_list: {len(criteria_list)}")
        unique_criteria_values = len(set(criteria_list))
        duplicate_values = len(criteria_list) - unique_criteria_values
        print(f"Number of unique values in criteria_list: {unique_criteria_values}")
        print(f"Number of duplicate values in criteria_list: {duplicate_values}")

        # Sort the criteria_list and get indices of the top uncertain samples
        uncertain_samples_indices = sorted(range(len(criteria_list)), key=lambda i: criteria_list[i], reverse=True)[
                                    :num_samples]
        print(f"Selected {len(uncertain_samples_indices)} uncertain samples indices")

        # Get the test data
        test_data = list(self.data.read_file(self.data.test_file))
        print(f"Read {len(test_data)} samples from test data")

        # Get the uncertain samples from the test data
        uncertain_samples = [test_data[i] for i in uncertain_samples_indices]

        # Step 1: Remove the selected uncertain samples from the test data
        for sample in uncertain_samples:
            test_data.remove(sample)

        # Step 2: Rewrite the test file without the selected uncertain samples
        with open(self.data.test_file, 'w', encoding="utf-8") as f:
            for lemma, word, tags in test_data:
                # Convert lists to strings if necessary
                if isinstance(lemma, list):
                    lemma = ''.join(lemma)
                if isinstance(word, list):
                    word = ''.join(word)
                if isinstance(tags, list):
                    tags = ';'.join(tags)
                f.write(f"{lemma}\t{word}\t{tags}\n")

        # Append the uncertain samples to the training file
        with open(self.data.train_file, 'a', encoding="utf-8") as f:
            for lemma, word, tags in uncertain_samples:
                # Convert lists to strings if necessary
                if isinstance(lemma, list):
                    lemma = ''.join(lemma)
                if isinstance(word, list):
                    word = ''.join(word)
                if isinstance(tags, list):
                    tags = ';'.join(tags)
                f.write(f"{lemma}\t{word}\t{tags}\n")

        # If the criteria is "information_density", update the kor_density.tsv to synchronize with the changes in the test pool
        if criteria == "information_density":
            # Remove the densities of the selected uncertain samples
            for i in uncertain_samples_indices:
                del density_values[i]

            # Rewrite the kor_density.tsv without the densities of the selected uncertain samples
            with open('kor_density.tsv', 'w', encoding="utf-8") as f:
                for lemma, density in zip([data[0] for data in test_data], density_values):
                    f.write(f"{lemma}\t{density}\n")

        print("Finished al_sampling")

    def random_sampling(self, num_samples=100):
        """
        Randomly select a given number of samples from the test set, append them to the training set,
        and remove them from the test set.

        Parameters:
        - num_samples (int): Number of samples to select.
        """
        # Get the test data
        test_data = list(self.data.read_file(self.data.test_file))

        # Randomly select samples from the test data
        random_samples = random.sample(test_data, num_samples)

        # Remove the selected samples from the test data
        for sample in random_samples:
            test_data.remove(sample)

        # Rewrite the test file without the selected samples
        with open(self.data.test_file, 'w', encoding="utf-8") as f:
            for lemma, word, tags in test_data:
                # Convert lists to strings if necessary
                if isinstance(lemma, list):
                    lemma = ''.join(lemma)
                if isinstance(word, list):
                    word = ''.join(word)
                if isinstance(tags, list):
                    tags = ';'.join(tags)
                f.write(f"{lemma}\t{word}\t{tags}\n")

        # Append the selected samples to the training file
        with open(self.data.train_file, 'a', encoding="utf-8") as f:
            for lemma, word, tags in random_samples:
                # Convert lists to strings if necessary
                if isinstance(lemma, list):
                    lemma = ''.join(lemma)
                if isinstance(word, list):
                    word = ''.join(word)
                if isinstance(tags, list):
                    tags = ';'.join(tags)
                f.write(f"{lemma}\t{word}\t{tags}\n")

    def select_model(self):
        best_res = [m for m in self.models if m.evaluation_result][0]
        best_acc = [m for m in self.models if m.evaluation_result][0]
        best_devloss = self.models[0]
        for m in self.models:
            if not m.evaluation_result:
                continue
            if (
                type(self.evaluator) == util.BasicEvaluator
                or type(self.evaluator) == util.PairBasicEvaluator
                or type(self.evaluator) == util.G2PEvaluator
                or type(self.evaluator) == util.PairG2PEvaluator
                or type(self.evaluator) == util.P2GEvaluator
                or type(self.evaluator) == util.HistnormEvaluator
            ):
                # [acc, edit distance / per ]
                if (
                    m.evaluation_result[0].res >= best_res.evaluation_result[0].res
                    and m.evaluation_result[1].res <= best_res.evaluation_result[1].res
                ):
                    best_res = m
            elif (
                type(self.evaluator) == util.TranslitEvaluator
                or type(self.evaluator) == util.PairTranslitEvaluator
            ):
                if (
                    m.evaluation_result[0].res >= best_res.evaluation_result[0].res
                    and m.evaluation_result[1].res >= best_res.evaluation_result[1].res
                ):
                    best_res = m
            else:
                raise NotImplementedError
            if m.evaluation_result[0].res >= best_acc.evaluation_result[0].res:
                best_acc = m
            if m.devloss <= best_devloss.devloss:
                best_devloss = m
        if self.params.bestacc:
            best_fp = best_acc.filepath
        else:
            best_fp = best_res.filepath
        return best_fp, set([best_fp])


def main():
    """
    main
    """
    trainer = Trainer()
    params = trainer.params
    trainer.params.decode_beam_size = 5

    decode_fn = get_decode_fn(
        params.decode, params.max_decode_len, params.decode_beam_size
    )
    trainer.load_data(params.dataset, params.train, params.dev, params.test)
    trainer.setup_evalutator()
    if params.load and params.load != "0":
        if params.load == "smart":
            start_epoch = trainer.smart_load_model(params.model) + 1
        else:
            start_epoch = trainer.load_model(params.load) + 1
        trainer.logger.info("continue training from epoch %d", start_epoch)
        trainer.setup_training()
        trainer.load_training(params.model)
    else:  # start from scratch
        start_epoch = 0
        trainer.build_model()
        if params.init:
            if os.path.isfile(params.init):
                trainer.load_state_dict(params.init)
            else:
                trainer.dump_state_dict(params.init)
        trainer.setup_training()

    trainer.run(start_epoch, decode_fn=decode_fn)


if __name__ == "__main__":
    main()
