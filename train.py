import torch
import torch.nn as nn
from evaluate import simple_evaluate,simple_evaluate_nomask,simple_evaluate_binary

def train(model:nn.Module,train_loader,optimiser,loss_fn:nn.Module,num_epochs:int,device:torch.device,test_loader=None,unimproved_epochs_max:int=5):
    scaler = torch.GradScaler()
    model.train()
    best_test_score = 0
    unimproved_epochs = 0

    for epoch in range(num_epochs):
        batch_num=0
        pos_distances = []
        neg_distances = []
        running_loss = 0.0


        for reference_data, query_data, labels, mask in train_loader:
            reference_data = reference_data.to(device)
            query_data = query_data.to(device)
            labels = labels.to(device).unsqueeze(-1)
            mask = mask.to(device).unsqueeze(-1)

            optimiser.zero_grad()

            with torch.autocast(device_type="cuda"):
                ref_embeddings, query_embeddings = model(reference_data, query_data, mask)
                loss = loss_fn(ref_embeddings.squeeze(),query_embeddings.squeeze(), labels.squeeze())

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            running_loss += loss.item()
            batch_num+=1
            with torch.no_grad():
                dist = torch.nn.functional.pairwise_distance(ref_embeddings.squeeze(), query_embeddings.squeeze())
                pos_dist = dist[labels.squeeze() == 1].mean().item()
                neg_dist = dist[labels.squeeze() == 0].mean().item()
                # if batch_num%10 == 0:
                #     print(f"Train Batch: {batch_num}")
                #     print(f"Train Batch Pos Dist: {pos_dist}")
                #     print(f"Train Batch Neg Dist: {neg_dist}")
                pos_distances.append(pos_dist)
                neg_distances.append(neg_dist)
                avg_loss = running_loss / len(train_loader)

        avg_pos = sum(pos_distances) / len(pos_distances)
        avg_neg = sum(neg_distances) / len(neg_distances)
        print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Pos Dist: {avg_pos:.4f} "
            f"Neg Dist: {avg_neg:.4f}")
        if test_loader is not None:
            total_correct,_ = simple_evaluate(model,test_loader,device)
            if total_correct > best_test_score:
                best_test_score = total_correct
                unimproved_epochs = 0
            else:
                unimproved_epochs +=1
            model.train()
        
        if unimproved_epochs>unimproved_epochs_max:
            break


def train_base(model:nn.Module,train_loader,optimiser,loss_fn:nn.Module,num_epochs:int,device:torch.device,test_loader=None,unimproved_epochs_max:int=5):
    scaler = torch.GradScaler()
    model.train()
    best_test_score = 0
    unimproved_epochs = 0

    for epoch in range(num_epochs):
        batch_num=0
        correct_predictions = 0
        total_predictions = 0
        running_loss = 0.0


        for reference_data, query_data, labels, mask in train_loader:
            reference_data = reference_data.to(device)
            query_data = query_data.to(device)
            labels = labels.to(device).unsqueeze(-1)
            mask = mask.to(device).unsqueeze(-1)

            optimiser.zero_grad()

            with torch.autocast(device_type="cuda"):
                predictions = model(reference_data, query_data, mask)
                loss = loss_fn(predictions.squeeze(), labels.squeeze())

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            running_loss += loss.item()
            batch_num+=1
            if batch_num%10==0:
                print(f"Batch {batch_num}/{len(train_loader)}")
            with torch.no_grad():
                rounded = (predictions>0.5).float()
                correct_predictions += torch.sum(rounded == labels)
                total_predictions += len(labels)

                avg_loss = running_loss / len(train_loader)

        avg = correct_predictions / total_predictions
        print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Acc: {avg:.4f}, {correct_predictions}/{total_predictions}")
        if test_loader is not None:
            total_correct = simple_evaluate_binary(model,test_loader,device)
            if total_correct > best_test_score:
                best_test_score = total_correct
                unimproved_epochs = 0
            else:
                unimproved_epochs +=1
            model.train()
        
        if unimproved_epochs>unimproved_epochs_max:
            break

    

def train_pairs(model:nn.Module,train_loader,optimiser,loss_fn:nn.Module,num_epochs:int,device:torch.device,test_loader=None):
    scaler = torch.GradScaler()
    model.train()
    best_test_score = 0
    unimproved_epochs = 0

    for epoch in range(num_epochs):
        batch_num=0
        pos_distances = []
        neg_distances = []
        running_loss = 0.0

        if test_loader is not None:
            total_correct,best_thresh = simple_evaluate_nomask(model,test_loader,device)
            if total_correct > best_test_score:
                best_test_score = total_correct
                unimproved_epochs = 0
            else:
                unimproved_epochs +=1
            model.train()


        for reference_data, query_data, labels in train_loader:
            reference_data = reference_data.to(device)
            query_data = query_data.to(device)
            labels = labels.to(device).unsqueeze(-1)

            optimiser.zero_grad()

            with torch.autocast(device_type="cuda"):
                ref_embeddings, query_embeddings = model(reference_data, query_data)
                loss = loss_fn(ref_embeddings.squeeze(),query_embeddings.squeeze(), labels.squeeze())
            # loss.backward()
            # optimiser.step()

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            running_loss += loss.item()
            batch_num+=1
            with torch.no_grad():
                dist = torch.nn.functional.pairwise_distance(ref_embeddings.squeeze(), query_embeddings.squeeze())
                pos_dist = dist[labels.squeeze() == 1].mean().item()
                neg_dist = dist[labels.squeeze() == 0].mean().item()
                # if batch_num%10 == 0:
                #     print(f"Train Batch: {batch_num}")
                #     print(f"Train Batch Pos Dist: {pos_dist}")
                #     print(f"Train Batch Neg Dist: {neg_dist}")
                pos_distances.append(pos_dist)
                neg_distances.append(neg_dist)
                avg_loss = running_loss / len(train_loader)

        avg_pos = sum(pos_distances) / len(pos_distances)
        avg_neg = sum(neg_distances) / len(neg_distances)
        print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Pos Dist: {avg_pos:.4f} "
            f"Neg Dist: {avg_neg:.4f}")

        
        if unimproved_epochs>50:
            break
