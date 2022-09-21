import os
import glob
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import optimizer
from copy import deepcopy
np.set_printoptions(threshold=np.inf)

from opacus import PrivacyEngine
from torch.optim import lr_scheduler
#from opacus.utils import module_modification
from sklearn.metrics import f1_score, roc_auc_score
#from opacus.dp_model_inspector import DPModelInspector

from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion.hop_skip_jump import HopSkipJump
import random
from attacks.utils.train import load_wo_module
from defenses.memguard import MemGuard
from opacus.validators.module_validator import ModuleValidator

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class shadow():
    def __init__(self, trainloader, testloader, model, device, use_DP, noise, norm, batch_size, loss, optimizer, nas_flag):
        self.use_DP = use_DP
        self.device = device
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        self.criterion = loss
        self.optimizer = optimizer
        self.nas_flag = nas_flag

        self.noise_multiplier, self.max_grad_norm = noise, norm
        
        if self.use_DP:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.trainloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
            print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))
            #privacy_engine.attach(self.optimizer)

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

    # Training
    def train(self):
        self.model.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            if self.nas_flag > 0:
                _, outputs = self.model(inputs)
            else:
                outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if self.use_DP:
            epsilon, best_alpha = self.privacy_engine.accountant.get_privacy_spent(delta=1e-5)
            #epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
            print("\u03B1: %.3f \u03B5: %.3f \u03B4: 1e-5" % (best_alpha, epsilon))
                
        self.scheduler.step()

        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return 1.*correct/total


    def saveModel(self, path):
        torch.save(self.model.state_dict(), path)

    def get_noise_norm(self):
        return self.noise_multiplier, self.max_grad_norm

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.nas_flag > 0:
                    _, outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total


class distillation_training():
    def __init__(self, PATH, trainloader, testloader, model, teacher, device, optimizer, T, alpha):
        self.device = device
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        self.PATH = PATH
        self.teacher = teacher.to(self.device)
        self.teacher.load_state_dict(torch.load(self.PATH))
        self.teacher.eval()

        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.optimizer = optimizer

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

        self.T = T
        self.alpha = alpha

    def distillation_loss(self, y, labels, teacher_scores, T, alpha):
        loss = self.criterion(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1))
        loss = loss * (T*T * alpha) + F.cross_entropy(y, labels) * (1. - alpha)
        return loss

    def train(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, [targets, _]) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            teacher_output = self.teacher(inputs)
            teacher_output = teacher_output.detach()
    
            loss = self.distillation_loss(outputs, targets, teacher_output, T=self.T, alpha=self.alpha)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

        self.scheduler.step()
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return 1.*correct/total

    def saveModel(self, path):
        torch.save(self.model.state_dict(), path)

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, [targets, _] in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total

class attack_for_blackbox():
    def __init__(self, SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, target_model, shadow_model, attack_model, device, nas_flag, use_memGuard):
        self.device = device

        self.TARGET_PATH = TARGET_PATH
        self.SHADOW_PATH = SHADOW_PATH
        self.ATTACK_SETS = ATTACK_SETS

        self.target_model = target_model.to(self.device)
        self.shadow_model = shadow_model.to(self.device)

        #self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
        #self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))
        load_wo_module(self.target_model,self.TARGET_PATH)
        load_wo_module(self.shadow_model,self.SHADOW_PATH)

        self.target_model.eval()
        self.shadow_model.eval()

        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_test_loader

        self.attack_model = attack_model.to(self.device)
        torch.manual_seed(0)
        self.attack_model.apply(weights_init)

        self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=5e-7)

        self.nas_flag=nas_flag
        self.use_memGuard=use_memGuard

    def _get_data(self, model, inputs, targets, use_nas, use_memGuard):
        if use_nas>0:
            _, result = model(inputs)
        else:
            result = model(inputs)
        if use_memGuard:
            memGuard=MemGuard()
            result=memGuard.forward(result)

        _, predicts = result.max(1)
        prediction = predicts.eq(targets).float().unsqueeze(1)
        
        output, _ = torch.sort(result, descending=True)

        return output.detach().cpu(), prediction.detach().cpu()

    def prepare_dataset(self):
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.shadow_model, inputs, targets, self.nas_flag-1, False)
                # output = output.cpu().detach().numpy()
            
                pickle.dump((output, prediction, members), f)

        print("Finished Saving Train Dataset")

        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.target_model, inputs, targets, self.nas_flag, self.use_memGuard)
                # output = output.cpu().detach().numpy()
            
                pickle.dump((output, prediction, members), f)

        print("Finished Saving Test Dataset")

    def train(self, epoch, result_path):
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0

        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while(True):
                try:
                    output, prediction, members = pickle.load(f)
                    output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

                    results = self.attack_model(output, prediction)
                    results = F.softmax(results, dim=1)

                    losses = self.criterion(results, members)
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    if epoch:
                        final_train_gndtrth.append(members)
                        final_train_predict.append(predicted)
                        final_train_probabe.append(results[:, 1])

                    batch_idx += 1
                except EOFError:
                    break

        if epoch:
            final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
            final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
            final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

            final_result.append(train_f1_score)
            final_result.append(train_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
            
            print("Saved Attack Train Ground Truth and Predict Sets")
            print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

        final_result.append(1.*correct/total)
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return final_result

    def test(self, epoch, result_path):
        self.attack_model.eval()
        batch_idx = 1
        correct = 0
        total = 0

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while(True):
                    try:
                        output, prediction, members = pickle.load(f)
                        output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

                        results = self.attack_model(output, prediction)
                        _, predicted = results.max(1)
                        total += members.size(0)
                        correct += predicted.eq(members).sum().item()
                        results = F.softmax(results, dim=1)

                        if epoch:
                            final_test_gndtrth.append(members)
                            final_test_predict.append(predicted)
                            final_test_probabe.append(results[:, 1])

                        batch_idx += 1
                    except EOFError:
                        break

        if epoch:
            final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
            final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
            final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

            test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
            test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

            final_result.append(test_f1_score)
            final_result.append(test_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

            print("Saved Attack Test Ground Truth and Predict Sets")
            print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

        final_result.append(1.*correct/total)
        print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return final_result

    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS +"test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)

class attack_for_whitebox():
    def __init__(self, TARGET_PATH, SHADOW_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, target_model, shadow_model, attack_model, device, class_num, nas_flag, use_memGuard):
        self.device = device
        self.class_num = class_num

        self.ATTACK_SETS = ATTACK_SETS

        self.TARGET_PATH = TARGET_PATH
        self.target_model = target_model.to(self.device)
        #self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
        load_wo_module(self.target_model,self.TARGET_PATH)
        self.target_model.eval()


        self.SHADOW_PATH = SHADOW_PATH
        self.shadow_model = shadow_model.to(self.device)
        #self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))
        load_wo_module(self.shadow_model,self.SHADOW_PATH)
        self.shadow_model.eval()

        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_test_loader

        self.attack_model = attack_model.to(self.device)
        torch.manual_seed(0)
        self.attack_model.apply(weights_init)

        self.target_criterion = nn.CrossEntropyLoss(reduction='none')
        self.attack_criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.SGD(self.attack_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)

        self.attack_train_data = None
        self.attack_test_data = None

        self.nas_flag=nas_flag
        self.use_memGuard=use_memGuard
        

    def _get_data(self, model, inputs, targets, use_nas, use_memGuard):
        if use_nas>0:
            _, results = model(inputs)
        else:
            results = model(inputs)
        if use_memGuard:
            memGuard=MemGuard()
            results=memGuard.forward(results)
        # outputs = F.softmax(outputs, dim=1)
        losses = self.target_criterion(results, targets)

        gradients = []
        
        for loss in losses:
            loss.backward(retain_graph=True)

            gradient_list = reversed(list(model.named_parameters()))

            for name, parameter in gradient_list:
                if 'weight' in name:
                    gradient = parameter.grad.clone() # [column[:, None], row].resize_(100,100)
                    while len(gradient.size())>2:
                        gradient.squeeze_(-1)
                    gradient = gradient.unsqueeze_(0)
                    gradients.append(gradient.unsqueeze_(0))
                    break

        labels = []
        for num in targets:
            label = [0 for i in range(self.class_num)]
            label[num.item()] = 1
            labels.append(label)

        gradients = torch.cat(gradients, dim=0)
        losses = losses.unsqueeze_(1).detach()
        outputs, _ = torch.sort(results, descending=True)
        labels = torch.Tensor(labels)

        return outputs.detach().cpu(), losses.cpu(), gradients.detach().cpu(), labels.cpu()

    def prepare_dataset(self):
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
                
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, loss, gradient, label = self._get_data(self.shadow_model, inputs, targets, self.nas_flag-1, False)

                pickle.dump((output, loss, gradient, label, members), f)

        print("Finished Saving Train Dataset")

        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, loss, gradient, label = self._get_data(self.target_model, inputs, targets, self.nas_flag, self.use_memGuard)
            
                pickle.dump((output, loss, gradient, label, members), f)

            # pickle.dump((output, loss, gradient, label, members), open(self.ATTACK_PATH + "test.p", "wb"))

        print("Finished Saving Test Dataset")

    
    def train(self, epoch, result_path):
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0

        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while(True):
                try:
                    output, loss, gradient, label, members = pickle.load(f)
                    output, loss, gradient, label, members = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device), members.to(self.device)

                    results = self.attack_model(output, loss, gradient, label)
                    # results = F.softmax(results, dim=1)
                    losses = self.attack_criterion(results, members)
                    
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    if epoch:
                        final_train_gndtrth.append(members)
                        final_train_predict.append(predicted)
                        final_train_probabe.append(results[:, 1])

                    batch_idx += 1
                except EOFError:
                    break	

        if epoch:
            final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
            final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
            final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

            final_result.append(train_f1_score)
            final_result.append(train_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
            
            print("Saved Attack Train Ground Truth and Predict Sets")
            print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

        final_result.append(1.*correct/total)
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return final_result


    def test(self, epoch, result_path):
        self.attack_model.eval()
        batch_idx = 1
        correct = 0
        total = 0

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while(True):
                    try:
                        output, loss, gradient, label, members = pickle.load(f)
                        output, loss, gradient, label, members = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device), members.to(self.device)

                        results = self.attack_model(output, loss, gradient, label)

                        _, predicted = results.max(1)
                        total += members.size(0)
                        correct += predicted.eq(members).sum().item()

                        results = F.softmax(results, dim=1)

                        if epoch:
                            final_test_gndtrth.append(members)
                            final_test_predict.append(predicted)
                            final_test_probabe.append(results[:, 1])

                        batch_idx += 1
                    except EOFError:
                        break

        if epoch:
            final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
            final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
            final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

            test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
            test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

            final_result.append(test_f1_score)
            final_result.append(test_roc_auc_score)


            with open(result_path, "wb") as f:
                pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

            print("Saved Attack Test Ground Truth and Predict Sets")
            print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

        final_result.append(1.*correct/total)
        print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return final_result

    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS +"test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)


def train_shadow_model(PATH, device, shadow_model, train_loader, test_loader, use_DP, noise, norm, batch_size, loss, optimizer, nas_flag, use_label_smooth=False):
    if use_DP:
        errors = ModuleValidator.validate(shadow_model, strict=False)
        if len(errors)>0:
            shadow_model=ModuleValidator.fix(shadow_model)
            optimizer = optim.SGD(shadow_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    _model=deepcopy(shadow_model)
    if use_label_smooth:
        loss = nn.CrossEntropyLoss(label_smoothing=1e-3)
    model = shadow(train_loader, test_loader, shadow_model, device, use_DP, noise, norm, batch_size, loss, optimizer, nas_flag)
    acc_train = 0
    acc_test = 0

    for i in range(100):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("shadow training")

        acc_train = model.train()
        print("shadow testing")
        acc_test = model.test()


        overfitting = round(acc_train - acc_test, 6)

        print('The overfitting rate is %s' % overfitting)

    FILE_PATH = PATH + "_shadow.pth"
    model.saveModel(FILE_PATH)
    print("saved shadow model!!!")
    print("Finished training!!!")

    return acc_train, acc_test, overfitting, _model

def train_shadow_distillation(MODEL_PATH, DL_PATH, device, target_model, student_model, train_loader, test_loader):
    distillation = distillation_training(MODEL_PATH, train_loader, test_loader, student_model, target_model, device)

    for i in range(100):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("shadow distillation training")

        acc_distillation_train = distillation.train()
        print("shadow distillation testing")
        acc_distillation_test = distillation.test()


        overfitting = round(acc_distillation_train - acc_distillation_test, 6)

        print('The overfitting rate is %s' % overfitting)

        
    result_path = DL_PATH + "_shadow.pth"

    distillation.saveModel(result_path)
    print("Saved shadow model!!!")
    print("Finished training!!!")

    return acc_distillation_train, acc_distillation_test, overfitting

def get_attack_dataset_without_shadow(train_set, test_set, batch_size):
    mem_length = len(train_set)//3
    nonmem_length = len(test_set)//3
    mem_train, mem_test, _ = torch.utils.data.random_split(train_set, [mem_length, mem_length, len(train_set)-(mem_length*2)])
    nonmem_train, nonmem_test, _ = torch.utils.data.random_split(test_set, [nonmem_length, nonmem_length, len(test_set)-(nonmem_length*2)])
    mem_train, mem_test, nonmem_train, nonmem_test = list(mem_train), list(mem_test), list(nonmem_train), list(nonmem_test)

    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_train)):
        nonmem_train[i] = nonmem_train[i] + (0,)
    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)
    for i in range(len(mem_test)):
        mem_test[i] = mem_test[i] + (1,)
        
    attack_train = mem_train + nonmem_train
    attack_test = mem_test + nonmem_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=batch_size, shuffle=True, num_workers=2)

    return attack_trainloader, attack_testloader

def get_attack_dataset_with_shadow(target_train, target_test, shadow_train, shadow_test, batch_size):
    mem_train, nonmem_train, mem_test, nonmem_test = list(shadow_train), list(shadow_test), list(target_train), list(target_test)

    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_train)):
        nonmem_train[i] = nonmem_train[i] + (0,)
    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)
    for i in range(len(mem_test)):
        mem_test[i] = mem_test[i] + (1,)


    train_length = min(len(mem_train), len(nonmem_train))
    test_length = min(len(mem_test), len(nonmem_test))

    mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
    non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])
    mem_test, _ = torch.utils.data.random_split(mem_test, [test_length, len(mem_test) - test_length])
    non_mem_test, _ = torch.utils.data.random_split(nonmem_test, [test_length, len(nonmem_test) - test_length])
    
    attack_train = mem_train + non_mem_train
    attack_test = mem_test + non_mem_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=batch_size, shuffle=True, num_workers=2)

    return attack_trainloader, attack_testloader


def attack_mode0(TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, get_attack_set, num_classes, nas_flag, use_memGuard=False):
    MODELS_PATH = ATTACK_PATH + "_meminf_attack0.pth"
    RESULT_PATH = ATTACK_PATH + "_meminf_attack0.p"
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode0_"


    attack = attack_for_blackbox(SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, device, nas_flag, use_memGuard)

    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()

    for i in range(50):
        flag = 1 if i == 49 else 0
        print("Epoch %d :" % (i+1))
        res_train = attack.train(flag, RESULT_PATH)
        res_test = attack.test(flag, RESULT_PATH)

    attack.saveModel(MODELS_PATH)
    print("Saved Attack Model")

    return res_train, res_test

def attack_mode1(TARGET_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model, attack_model, get_attack_set, num_classes, nas_flag, use_memGuard=False):
    MODELS_PATH = ATTACK_PATH + "_meminf_attack1.pth"
    RESULT_PATH = ATTACK_PATH + "_meminf_attack1.p"
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode1_"

    attack = attack_for_blackbox(TARGET_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, target_model, target_model, attack_model, device, nas_flag, use_memGuard)

    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()

    for i in range(50):
        flag = 1 if i == 49 else 0
        print("Epoch %d :" % (i+1))
        res_train = attack.train(flag, RESULT_PATH)
        res_test = attack.test(flag, RESULT_PATH)

    attack.saveModel(MODELS_PATH)
    print("Saved Attack Model")

    return res_train, res_test

def attack_mode2(TARGET_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model, attack_model, get_attack_set, num_classes, nas_flag, use_memGuard=False):
    MODELS_PATH = ATTACK_PATH + "_meminf_attack2.pth"
    RESULT_PATH = ATTACK_PATH + "_meminf_attack2.p"
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode2_"

    attack = attack_for_whitebox(TARGET_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, target_model, target_model, attack_model, device, num_classes, nas_flag, use_memGuard)
    
    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()

    for i in range(50):
        flag = 1 if i == 49 else 0
        print("Epoch %d :" % (i+1))
        res_train = attack.train(flag, RESULT_PATH)
        res_test = attack.test(flag, RESULT_PATH)

    attack.saveModel(MODELS_PATH)
    print("Saved Attack Model")

    return res_train, res_test

def attack_mode3(TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, get_attack_set, num_classes, nas_flag, use_memGuard=False):
    MODELS_PATH = ATTACK_PATH + "_meminf_attack3.pth"
    RESULT_PATH = ATTACK_PATH + "_meminf_attack3.p"
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode3_"

    attack = attack_for_whitebox(TARGET_PATH, SHADOW_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, device, num_classes, nas_flag, use_memGuard)
    
    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()

    for i in range(50):
        flag = 1 if i == 49 else 0
        print("Epoch %d :" % (i+1))
        res_train = attack.train(flag, RESULT_PATH)
        res_test = attack.test(flag, RESULT_PATH)

    attack.saveModel(MODELS_PATH)
    print("Saved Attack Model")

    return res_train, res_test

def get_gradient_size(model):
    gradient_size = []
    gradient_list = reversed(list(model.named_parameters()))
    for name, parameter in gradient_list:
        if 'weight' in name:
            gradient_size.append(parameter.shape)

    return gradient_size

class NAS2normal(nn.Module):
    def __init__(self,net):
        super(NAS2normal, self).__init__()
        self.net=net
    def forward(self, x):
        _, results=self.net(x)
        return results

def label_only_attack(estimator, x_train, y_train, x_test, y_test, sample_size, xshape, num_classes, seed, is_nas):
    print('x_train.min(): {}, x_train.max(): {}'.format(x_train.min(),x_train.max()))
    print('x_test.min(): {}, x_test.max(): {}'.format(x_test.min(),x_test.max()))
    low=min(x_train.min(),x_test.min())
    high=max(x_train.max(),x_test.max())
    print("low: {}, high: {}".format(low,high))

    if is_nas:
        classifier=PyTorchClassifier(
            model=NAS2normal(estimator),
            loss = nn.CrossEntropyLoss(),
            optimizer=optim.SGD(estimator.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4),
            input_shape=xshape,
            nb_classes=num_classes,
            clip_values=(low, high)
        )
    else:
        classifier=PyTorchClassifier(
            model=estimator,
            loss = nn.CrossEntropyLoss(),
            optimizer=optim.SGD(estimator.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4),
            input_shape=xshape,
            nb_classes=num_classes,
            clip_values=(low, high)
        )


    train_indices=list(range(len(x_train)))
    test_indices=list(range(len(x_test)))

    np.random.seed(seed)
    random_seeds=np.random.randint(1,1000000,2)
    random.seed(random_seeds[0])
    random.shuffle(train_indices)
    random.seed(random_seeds[1])
    random.shuffle(test_indices)

    attack_train_size=sample_size
    attack_test_size=sample_size

    #max_iter = 10
    #max_eval = 100
    #init_size = 50
    #init_eval = 25

    max_iter = 50
    max_eval = 2500
    init_size = 100
    init_eval = 100

    print("max_iter: {}, max_eval: {}, init_size: {}, init_eval: {}, sample_size: {}".format(max_iter,max_eval,init_size,init_eval, sample_size))

    hsj = HopSkipJump(classifier=classifier, targeted=False, norm=2, max_iter=max_iter,
                      init_size=init_size, init_eval=init_eval, max_eval=max_eval, verbose=True)

    x=np.concatenate((x_train[train_indices[:attack_train_size]],x_test[test_indices[:attack_test_size]]))
    y_truth=np.concatenate((np.ones(attack_train_size),np.zeros(attack_test_size)),axis=0)

    # Note: there might be some rare cases when the HopSkipJump attack failed and the original images would be returned
    x_adv = hsj.generate(x=x, y=None)
    distance = np.linalg.norm((x_adv - x).reshape((x.shape[0], -1)), ord=np.inf, axis=1)

    print("\nOriginal distance:\n{}".format(list(distance)))
    auc = roc_auc_score(y_truth, distance)
    print("Original Label-Only Membership Inference Attack AUC: {:4f}".format(auc))

    # Failed attack means it is **hard** to change the predicted label of the original image with the limited perturbations.
    # The distance for these failed attacks should be large enough to show that attcking the corresponding images needs a stronger attack.
    distance[distance==0]=distance.max()+(distance.max()-distance.min())/100
    print("\nModified distance:\n{}".format(list(distance)))

    auc = roc_auc_score(y_truth, distance)
    print("Label-Only Membership Inference Attack AUC: {:4f}".format(auc))