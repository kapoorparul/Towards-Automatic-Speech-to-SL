import argparse
import time
import shutil
import os
import queue
import numpy as np

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset

from model import build_model
from batch import Batch
from helpers import  load_config, log_cfg, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError, get_latest_checkpoint
from model import Model
from prediction import validate_on_data
from loss import RegLoss, XentLoss
from data import load_data, make_data_iter
from builders import build_optimizer, build_scheduler, \
    build_gradient_clipper
from constants import TARGET_PAD
from torch.autograd import Variable


from plot_videos import plot_video,alter_DTW_timing

from discriminator import Classifier, ClassifierLayers


## add padding till max length
def padding_tensor(sequences, max_len):
    """
    :param sequences: list of tensors
    :return: padded tensor
    """
    num = len(sequences)
    
    out_dims = (num, max_len, sequences.shape[-1])
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        
    return out_tensor

class TrainManager:

    def __init__(self, model: Model, disc: Classifier, config: dict, test=False) -> None:

        train_config = config["training"]
        model_dir = train_config["model_dir"]
        # If model continue, continues model from the latest checkpoint
        model_continue = True if not test else False #train_config.get("continue", True)
        # If the directory has not been created, can't continue from anything
        if not os.path.isdir(model_dir):
            model_continue = False
        if test:
            model_continue = True

        # files for logging and storing
        self.model_dir = make_model_dir(train_config["model_dir"],
                                        overwrite=train_config.get("overwrite", False),
                                        model_continue=model_continue)
        # Build logger
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        
        # Build validation files
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir+"/tensorboard/")

        # model
        self.model = model
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self.nonreg_pad_index = self.model.nonreg_pad_index

        self._log_parameters_list()
        self.target_pad = TARGET_PAD

        # New Regression loss - depending on config
        self.loss = RegLoss(cfg = config,
                            target_pad=self.target_pad)
        self.nonreg_loss_function = XentLoss(pad_index=self.nonreg_pad_index)
        self.regloss_weight=train_config.get("regloss_weight", 1)
        self.xentloss_weight=train_config.get("xentloss_weight",0)
        self.advloss_weight = train_config.get("advloss_weight",0)
        self.normalization = "batch"

        self.disc = disc
        self.disc_opt = build_optimizer(config=config["training"]["disc"], parameters=self.disc.parameters())
        

        self.src_fps = config["data"]["src_fps"]
        self.trg_fps = config["data"]["trg_fps"]
        self.num_sec = config["data"]["num_sec"]


        # optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config, parameters=model.parameters())

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.ckpt_best_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 1))
        self.ckpt_queue = queue.Queue(maxsize=1)

        self.val_on_train = config["data"].get("val_on_train", False)

        
        self.eval_metric = train_config.get("eval_metric", "dtw").lower()
        self.early_stopping_metric = train_config.get("early_stopping_metric",
                                                       "eval_metric")

        self.minimize_metric = True
        

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        # data & batch handling
        self.level = "word" ## or char
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = "sentence"
        self.eval_batch_size = train_config.get("eval_batch_size",self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type",self.batch_type)
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # generation
        self.max_output_length = train_config.get("max_output_length", None)

        # CPU / GPU
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            self.loss.cuda()
            self.disc.cuda()

        # initialize training statistics
        self.steps = 0
        
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_iteration = 0
        
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        
        # comparison function for scores
        self.is_best = lambda score: score < self.best_ckpt_score \
            if self.minimize_metric else score > self.best_ckpt_score

        ## Checkpoint restart
        ## If continuing
        if model_continue:
            # Get the latest checkpoint
            ckpt = get_latest_checkpoint(model_dir)
            if ckpt is None:
                self.logger.info("Can't find checkpoint in directory %s", ckpt)
            else:
                self.logger.info("Continuing model from %s", ckpt)
                self.init_from_checkpoint(ckpt)

        # Skip frames
        self.skip_frames = config["data"].get("skip_frames", 1)

        ## -- Data augmentation --
        # Just Counter
        self.just_count_in = config["model"].get("just_count_in",False)
        # Gaussian Noise
        self.gaussian_noise = config["model"].get("gaussian_noise", False)
        if self.gaussian_noise:
            # How much the noise is added in
            self.noise_rate = config["model"].get("noise_rate", 1.0)
        else:
            self.noise_rate = 0.0
        if self.just_count_in and (self.gaussian_noise):
            raise ConfigurationError("Can't have both just_count_in and gaussian_noise as True")

        self.future_prediction = config["model"].get("future_prediction", 0)
        if self.future_prediction != 0:
            frames_predicted = [i for i in range(self.future_prediction)]
            self.logger.info("Future prediction. Frames predicted: %s",frames_predicted)

    # Save a checkpoint
    def _save_checkpoint(self, type="every") -> None:
        # Define model path
        model_path = "{}/{}_{}.ckpt".format(self.model_dir, self.steps, type)
        # Define State
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if \
            self.scheduler is not None else None,
            "disc_state":self.disc.state_dict(),
            "disc_opt_state":self.disc_opt.state_dict()
        }
        torch.save(state, model_path)
        # If this is the best checkpoint
        if type == "best":
            if self.ckpt_best_queue.full():
                to_delete = self.ckpt_best_queue.get()  # delete oldest ckpt
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but "
                                        "file does not exist.", to_delete)

            self.ckpt_best_queue.put(model_path)

            best_path = "{}/best.ckpt".format(self.model_dir)
            try:
                # create/modify symbolic link for best checkpoint
                symlink_update("{}_best.ckpt".format(self.steps), best_path)
            except OSError:
                # overwrite best.ckpt
                torch.save(state, best_path)

        # If this is just the checkpoint at every validation
        elif type == "every":
            if self.ckpt_queue.full():
                to_delete = self.ckpt_queue.get()  # delete oldest ckpt
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but "
                                        "file does not exist.", to_delete)

            self.ckpt_queue.put(model_path)

            every_path = "{}/every.ckpt".format(self.model_dir)
            try:
                # create/modify symbolic link for best checkpoint
                symlink_update("{}_best.ckpt".format(self.steps), every_path)
            except OSError:
                # overwrite every.ckpt
                torch.save(state, every_path)

    # Initialise from a checkpoint
    def init_from_checkpoint(self, path: str) -> None:
        # Find last checkpoint
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])
        self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])

        self.disc.load_state_dict(model_checkpoint["disc_state"])
        self.disc_opt.load_state_dict(model_checkpoint["disc_opt_state"])

        if model_checkpoint["scheduler_state"] is not None and \
                self.scheduler is not None:
            # Load the scheduler state
            self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]
        self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
        self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()
            self.disc.cuda()

    # Train and validate function
    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) \
            -> None:
        # Make training iterator
        train_iter = make_data_iter(train_data,
                                    batch_size=self.batch_size,
                                    batch_type=self.batch_type,
                                    train=True, shuffle=self.shuffle)

        val_step = 0
        if self.gaussian_noise:
            all_epoch_noise = []

        # all_epoch_noise = []
        # Loop through epochs
        for epoch_no in range(self.epochs):
                
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()

            # Reset statistics for each epoch.
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.total_tokens
            count = self.batch_multiplier - 1
            epoch_loss = 0

            # If Gaussian Noise, extract STDs for each joint position
            if self.gaussian_noise:
                if len(all_epoch_noise) != 0:
                    self.model.out_stds = torch.mean(torch.stack(([noise.std(dim=[0]) for noise in all_epoch_noise])),dim=-2)
                else:
                    self.model.out_stds = None
                all_epoch_noise = []

            for batch in iter(train_iter):
                # reactivate training
                self.model.train()

                # create a Batch object from torchtext batch
                batch = Batch(torch_batch=batch,
                              pad_index=self.pad_index,
                              nonreg_pad_index = self.nonreg_pad_index,
                              model=self.model)

                update = count == 0
                # Train the model on a batch
                batch_loss, noise, skel_out = self._train_batch(batch, update=update, noise_rate=self.noise_rate)
                
                ############################################################
                ###################################### calculate discriminator loss
                    
                seq_valid = Variable(Tensor(np.ones((batch.trg.shape[0], 1))),requires_grad=False)
                seq_fake_gt = Variable(Tensor(np.zeros((batch.trg.shape[0], 1))),requires_grad=False)
                
                if self.use_cuda:
                    seq_valid = seq_valid.to('cuda')
                    seq_fake_gt = seq_fake_gt.to('cuda')
                
                adversarial_loss = torch.nn.BCEWithLogitsLoss() 
                
                if self.future_prediction != 0:
                    # Cut to only the first frame prediction
                    out = torch.cat((skel_out[:, :, :skel_out.shape[2] // (self.future_prediction)],skel_out[:,:,-1:]),dim=2)
                    batch_out = torch.cat((batch.trg[:, :, :batch.trg.shape[2] // (self.future_prediction)],batch.trg[:,:,-1:]),dim=2)
                else:
                    batch_out = batch.trg
                
                if self.future_prediction != 0:
                    batch_out = padding_tensor(batch_out, out.shape[1] + self.future_prediction + 1)
                    out = padding_tensor(out, out.shape[1] + self.future_prediction + 1)
                
                ############# when all batches are of different lengths, not needed when input is cropped audio/vid
                src = padding_tensor(batch.src, self.num_sec*self.src_fps)
                out = padding_tensor(out, self.num_sec*self.trg_fps)
                batch_out = padding_tensor(batch_out, self.num_sec*self.trg_fps)

                seq_fake = self.disc(src, out[:, :, :-1])
                
                disc_loss = adversarial_loss(seq_fake, seq_valid)

                
                # normalize batch loss
                if self.normalization == "batch":
                    normalizer = batch.nseqs
                elif self.normalization == "tokens":
                    normalizer = batch.ntokens
                else:
                    raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

                norm_batch_loss = batch_loss / normalizer
                
                # division needed since loss.backward sums the gradients until updated
                norm_batch_multiply = norm_batch_loss / self.batch_multiplier

                disc_loss = disc_loss / normalizer
                
                norm_batch_multiply +=  self.advloss_weight*disc_loss
                
                self.optimizer.zero_grad()

                # compute gradients
                norm_batch_multiply.backward()

                if self.clip_grad_fun is not None:
                    # clip gradients (in-place)
                    self.clip_grad_fun(params=self.model.parameters())


                ### train model, make gradient step
                if update:
                    self.optimizer.step()
                    

                    # increment step counter
                    self.steps += 1

                # increment token counter
                self.total_tokens += batch.ntokens

                #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                # -----------------------------------------
                #         Train Discriminator 
                # ---------------------------------------

                self.disc.zero_grad()
                
                # Real loss
                pred_real_seq = self.disc(src, batch_out[:, :, :-1])
                loss_real_seq = adversarial_loss(pred_real_seq, seq_valid)

                # Fake loss
                pred_fake_seq = self.disc(src, out[:, :, :-1].detach())
                loss_fake_seq = adversarial_loss(pred_fake_seq, seq_fake_gt)
                
                # Total loss for discriminator
                if update:
                    D_loss_seq = (0.5 * (loss_real_seq + loss_fake_seq)) 
                    loss_D = D_loss_seq
                    loss_D.backward()
                    self.disc_opt.step()

                #####################################################################
                


                # If Gaussian Noise, collect the noise
                if self.gaussian_noise:
                    # If future Prediction, cut down the noise size to just one frame
                    if self.future_prediction != 0:
                        all_epoch_noise.append(noise.reshape(-1, self.model.out_trg_size // self.future_prediction))
                    else:
                        all_epoch_noise.append(noise.reshape(-1,self.model.out_trg_size))

                self.tb_writer.add_scalar("train/train_batch_loss", batch_loss,self.steps)
                count = self.batch_multiplier if update else count
                count -= 1
                epoch_loss += batch_loss.detach().cpu().numpy()

                if self.scheduler is not None and self.scheduler_step_at == "step" and update:
                    self.scheduler.step()

                # log learning progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - start_tokens
                    self.logger.info(
                        "Epoch %3d Step: %8d Batch Loss: %12.6f "
                        "Tokens per Sec: %8.0f, Lr: %.6f",
                        epoch_no + 1, self.steps, batch_loss,
                        elapsed_tokens / elapsed,
                        self.optimizer.param_groups[0]["lr"])
                    start = time.time()
                    total_valid_duration = 0
                    start_tokens = self.total_tokens

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:

                    valid_start_time = time.time()

                    valid_score, valid_loss, valid_references, valid_hypotheses, \
                        valid_inputs, all_dtw_scores, valid_file_paths, pck = \
                        validate_on_data(
                            batch_size=self.eval_batch_size,
                            data=valid_data,
                            eval_metric=self.eval_metric,
                            model=self.model,
                            max_output_length=self.max_output_length,
                            loss_function=self.loss,
                            nonreg_loss_function=self.nonreg_loss_function,
                            batch_type=self.eval_batch_type,
                            type="val",
                        )

                    val_step += 1

                    # Tensorboard writer
                    self.tb_writer.add_scalar("valid/valid_loss", valid_loss, self.steps)
                    self.tb_writer.add_scalar("valid/valid_score", valid_score, self.steps)

                    if self.early_stopping_metric == "loss":
                        ckpt_score = valid_loss
                    elif self.early_stopping_metric == "dtw":
                        ckpt_score = valid_score
                    else:
                        ckpt_score = valid_score

                    new_best = False
                    self.best = False
                    if self.is_best(ckpt_score):
                        self.best = True
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            'Hooray! New best validation result [%s]!',
                            self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint(type="best")

                        # Display these sequences, in this index order
                        display = list(range(0, len(valid_hypotheses), int(np.ceil(len(valid_hypotheses) / 10))))

                        self.produce_validation_video(
                            output_joints=valid_hypotheses,
                            inputs=valid_inputs,
                            references=valid_references,
                            model_dir=self.model_dir,
                            steps=self.steps,
                            display=display,
                            type="val_inf",
                            file_paths=valid_file_paths,
                        )
                        
                    self._save_checkpoint(type="every")

                    if self.scheduler is not None and self.scheduler_step_at == "validation":
                        self.scheduler.step(ckpt_score)

                    # append to validation report
                    self._add_report(
                        valid_score=valid_score, valid_loss=valid_loss,
                        eval_metric=self.eval_metric,
                        new_best=new_best, report_type="val",)

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        'Validation result at epoch %3d, step %8d: Val DTW Score: %6.2f, '
                        'loss: %8.4f, PCK %8.4f, duration: %.4fs',
                            epoch_no+1, self.steps, valid_score,
                            valid_loss, np.mean(pck), valid_duration)

                if self.stop:
                    break
            if self.stop:
                self.logger.info(
                    'Training ended since minimum lr %f was reached.',
                     self.learning_rate_min)
                break

            self.logger.info('Epoch %3d: total training loss %.5f', epoch_no+1,
                             epoch_loss)
        else:
            self.logger.info('Training ended after %3d epochs.', epoch_no+1)
        self.logger.info('Best validation result at step %8d: %6.2f %s.',
                         self.best_ckpt_iteration, self.best_ckpt_score,
                         self.early_stopping_metric)

        self.tb_writer.close()  # close Tensorboard writer

    # Produce the video 
    def produce_validation_video(self,output_joints, inputs, references, display, model_dir, type, steps="", file_paths=None):

        # If not at test
        if type != "test":
            dir_name = model_dir + "/videos/Step_{}/".format(steps)
            if not os.path.exists(model_dir + "/videos/"):
                os.mkdir(model_dir + "/videos/")

        # If at test time
        elif type == "test":
            dir_name = model_dir + "/test_videos/"

        # Create model video folder if not exist
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        # For sequence to display
        for i in display:

            seq = output_joints[i]
            ref_seq = references[i]
            
            input = inputs[i]

            # Write Video name
            gloss_label = file_paths[i] 
            
            # Alter the dtw timing of the produced sequence, and collect the DTW score
            timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(seq, ref_seq)
            

            video_ext = "{}_{}.mp4".format(gloss_label, "{0:.2f}".format(float(dtw_score)).replace(".", "_"))
            
            if file_paths is not None:
                sequence_ID = file_paths[i]
            else:
                sequence_ID = None

            variation = gloss_label.split('_')[-1]
            audio_name = "" 
            for a in gloss_label.split('_')[:-1]:
                audio_name += a
                audio_name += '_'

            ## Add audio path :: Change this according to audio file location
            audio_path = os.path.join("../../Data/", audio_name[:-1], 'audio', variation+'.wav') 

            # Plot this sequences video
            if "<" not in video_ext:
                plot_video(joints=timing_hyp_seq,
                           file_path=dir_name,
                           video_name=video_ext,
                           references=ref_seq_count,
                           skip_frames=self.skip_frames,
                           sequence_ID=sequence_ID,
                           audio_path=audio_path)

    # Train the batch
    def _train_batch(self, batch: Batch, update: bool = True, noise_rate:int =0) -> Tensor:

        # Get loss from this batch
        batch_loss, noise, skel_out = self.model.get_loss_for_batch(
            batch=batch, loss_function=self.loss, nonreg_loss_function=self.nonreg_loss_function, 
            xentloss_weight=self.xentloss_weight, regloss_weight=self.regloss_weight,noise_rate=noise_rate)

        return batch_loss, noise, skel_out


    def _add_report(self, valid_score: float, valid_loss: float, eval_metric: str,
                    new_best: bool = False, report_type: str = "val") -> None:

        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']


        if report_type == "val":
            with open(self.valid_report_file, 'a') as opened_file:
                opened_file.write(
                    "Steps: {} Loss: {:.5f}| DTW: {:.3f}|"
                    " LR: {:.6f} {}\n".format(
                        self.steps, valid_loss, valid_score,
                        current_lr, "*" if new_best else ""))

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad,
                                  self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [n for (n, p) in self.model.named_parameters()
                            if p.requires_grad]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params


def train(cfg_file: str) -> None:

    # Load the config file
    cfg = load_config(cfg_file)

    # Set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # Load the data - Trg as (batch, # of frames, #joints + 1 )
    train_data, dev_data, test_data, src_vocab, trg_vocab, nonreg_trg_vocab = load_data(cfg=cfg)
    
    # Build the  Transformer model and Cross modal discriminator
    model, disc = build_model(cfg, src_vocab=src_vocab, trg_vocab=trg_vocab, nonreg_trg_vocab=nonreg_trg_vocab)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, disc=disc, config=cfg)

    # Store copy of original training config in model dir
    shutil.copy2(cfg_file, trainer.model_dir+"/config.yaml")

    # Log all entries of config
    log_cfg(cfg, trainer.logger)

    # Train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    # Test the model with the best checkpoint
    test(cfg_file)

# pylint: disable-msg=logging-too-many-args
def test(cfg_file,
         ckpt: str=None) -> None:

    # Load the config file
    cfg = load_config(cfg_file)

    # Load the model directory and checkpoint
    model_dir = cfg["training"]["model_dir"]
    
    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir,post_fix="_best")
        if ckpt is None:
            raise FileNotFoundError("No checkpoint found in directory {}."
                                    .format(model_dir))

    batch_size = cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
    batch_type = cfg["training"].get("eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", False)
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab, nonreg_trg_vocab = load_data(cfg=cfg,mode = 'test')

    # To produce testing results
    data_to_predict = {"test": test_data}
    # To produce validation results
    # data_to_predict = {"dev": dev_data}

    # Load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)
    print("LOADED MODEL ........... ", ckpt)

    # Build model, discriminator and load parameters into model
    model, disc = build_model(cfg, src_vocab=src_vocab, trg_vocab=trg_vocab, nonreg_trg_vocab=nonreg_trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    # If cuda, set model as cuda
    if use_cuda:
        model.cuda()

    # Set up trainer to produce videos
    trainer = TrainManager(model=model, disc=disc,config=cfg, test=True)
    all_dtws = []
    all_pck = []

    # For each of the data batch, produce results
    for data_set_name, data_set in data_to_predict.items():

        # Validate for this data set
        score, loss, references, hypotheses, \
        inputs, all_dtw_scores, file_paths, pck = \
            validate_on_data(
                model=model,
                data=data_set,
                batch_size=batch_size,
                max_output_length=max_output_length,
                eval_metric=eval_metric,
                loss_function=None,
                batch_type=batch_type,
                type="val" if not data_set_name is "train" else "train_inf"
            )
        all_dtws.append(all_dtw_scores)
        all_pck.append(pck)
        
        # Set which sequences to produce video for
        display = list(range(len(hypotheses)))
        
        # # Produce videos for the produced hypotheses
        trainer.produce_validation_video(
            output_joints=hypotheses,
            inputs=inputs,
            references=references,
            model_dir=model_dir,
            display=display,
            type="test",
            file_paths=file_paths,
        )
    print("Mean dtw : ", np.mean(all_dtws))
    print("Mean pck : ", np.mean(all_pck))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Progressive Transformers")
    parser.add_argument("config", default="configs/default.yaml", type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)


