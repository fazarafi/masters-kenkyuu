import pandas
import numpy as np
import torch
import train_abstractive from ../PreSumm

import Decoder from 

class SummModel(object):
  """Class of Summarization model. Supports all baseline models"""
  def __init__(self):
        super(SummModel, self).__init__()
        self.factcc_rewards = []
        self.theta_param = []

  def calculate_reward(self, reference, summary, measurement):
    # Calculate reward as for 
    if 'factcc' in measurement:
      return self.factcc_cls(reference, summary)
    if 'feqa' in measurement:
      return self.feqa_cls(reference, summary)
    else: 
      return "No evaluation defined"

  def factcc_cls(self, ref, summary):
    # FactCC Eval Method (Krysynci, 2019)
    # Feed reference and summary to classification model
    if (ref==summary):
      return self.factcc_rewards.append(1)
    else:
      return self.factcc_rewards.append(0)


  def feqa_cls(self, ref, summary):
    # FEQA Eval Method ()
    # Feed reference and summary to classification model
    if (ref==summary):
      return 1
    else:
      return 0

  def max_likelihood():
    # Critic I

  def loss_function(params):
    return 0
    
  def loss_function_val(params):
    return 0

  def rl_training(self, summ_data):
    # Training of reinforcement learining network during using decoding part
    # Actor-critic algorithm
    
    # Parameters initialization
    k_all = []
    theta = 0
    pi = 0
    a_theta = 0.5
    a_pi = 0.5

    step = 3

    cnvrgd = False
    while (not(cnvrgd)):
      # Pretraining with Critic I
      

      for i in k_all:
        # Every k-step of optimization
        if i % step == 0:


        # Critic I optimization
        theta = theta - a_theta * self.loss_function(theta)

        # Critic II optimization
        pi = pi - a_pi * self.loss_function_val(summ_data)

        







