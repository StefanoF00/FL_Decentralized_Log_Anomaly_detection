import torch
import numpy as np
from pinns.loss import compute_loss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json
'''
def train(data):  
    name = data.get("name", "main")
    model = data.get("model")
    epochs = data.get("epochs")
    batchsize = data.get("batchsize")
    optimizer = data.get("optimizer")
    scheduler_step = data.get("scheduler_step")
    scheduler_reduce = data.get("scheduler_reduce")
    ode_fn = data.get("ode_fn")
    domaindataset = data.get("domain_dataset")
    validationdomaindataset = data.get("validation_domain_dataset")
    Q = data.get("Q") 
    positive_w = data.get("positive_w")
    derivative_w = data.get("derivative_w")
    properness_w = data.get("properness_w")
    normalization = data.get("normalization")
    if scheduler_step!=None and scheduler_reduce!=None:
        raise ValueError('Both schedulers were given.')
        
    scheduler = None
    if scheduler_step!=None:
        scheduler = scheduler_step
    if scheduler_reduce!=None:
        scheduler = scheduler_reduce
    current_file = os.getcwd()
    sys_dir = os.path.join(current_file, "output", name)
    output_dir = os.path.join(sys_dir, name+"_01")
    
    if os.path.exists(output_dir):
        counter = 2
        while True:
            i=str(counter)
            if counter<10:
                i = "0"+i
            output_dir = os.path.join(sys_dir, name+"_"+i)
            if not os.path.exists(output_dir):
                break
            else:
                counter +=1
                
    model_dir = os.path.join(output_dir, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, f"model.pt")
    file_path = f"{output_dir}/train.txt"

    params_path = f"{output_dir}/params.json"
    params = {
        "name": name,
        "model": str(model),
        "epochs": epochs,
        "batchsize": batchsize,
        "optimizer": str(optimizer),
        "scheduler": str(scheduler.state_dict()) if scheduler != None else "None",
        "domainDataset": str(domaindataset),
        "validationDomainDataset": str(validationdomaindataset)
    } 
    with open(params_path, 'w') as fp:
        json.dump(params, fp)
    
    domain_dataloader = DataLoader(domaindataset, batch_size=batchsize, shuffle=True)
    validation_dataloader = DataLoader(validationdomaindataset, batch_size=batchsize, shuffle=False)

    train_losses = []  # To store training losses
    validation_losses = []  # To store validation losses
    validation_losses_pos = []
    validation_losses_der = []
    validation_losses_prop = []
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning rate: {current_lr}")
    with open(file_path, 'w') as log_file:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = []
            epoch_loss_pos = []
            epoch_loss_der = []
            epoch_loss_prop = []
            for batch_idx, x_in in enumerate(domain_dataloader):
                optimizer.zero_grad()
                x_in = torch.Tensor(x_in).to(torch.device('cuda:0'))
                loss_positive, loss_derivative, loss_properness = compute_loss(
                    normalization,\
                    x_in,\
                    model,\
                    ode_fn,\
                    Q,\
                    positive_w,\
                    derivative_w,\
                    properness_w
                )
                loss = loss_positive + loss_derivative + loss_properness
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                epoch_loss_pos.append(loss_positive.item())
                epoch_loss_der.append(loss_derivative.item())
                epoch_loss_prop.append(loss_properness.item())

            avg_train_loss = np.mean(epoch_loss)
            avg_train_loss_pos = np.mean(epoch_loss_pos)
            avg_train_loss_der = np.mean(epoch_loss_der)
            avg_train_loss_prop = np.mean(epoch_loss_prop)
            train_losses.append(avg_train_loss)
            print(f'Epoch {epoch} \t Train Loss: {avg_train_loss:.5f};', end= ' \t ')
            log_file.write(f'Epoch {epoch} \t Training Loss: {avg_train_loss:.10f}\n')

            # Validation and learning rate adjustment every 10 valids
            model.eval()
            validation_loss = []
            valid_loss_pos = []
            valid_loss_der = []
            valid_loss_prop = []
            for batch_idx, x_in in enumerate(validation_dataloader):
                x_in = torch.Tensor(x_in).to(torch.device('cuda:0'))
                val_loss_pos, val_loss_der, val_loss_prop = compute_loss(
                    normalization,\
                    x_in,\
                    model,\
                    ode_fn,\
                    Q,\
                    positive_w,\
                    derivative_w,\
                    properness_w
                )
                val_loss = val_loss_pos + val_loss_der + val_loss_prop
                validation_loss.append(val_loss.item())
                valid_loss_pos.append(val_loss_pos.item())
                valid_loss_der.append(val_loss_der.item())
                valid_loss_prop.append(val_loss_prop.item())
                
            avg_validation_loss = np.average(validation_loss)
            avg_valid_loss_pos = np.average(valid_loss_pos)
            avg_valid_loss_der = np.average(valid_loss_der)
            avg_valid_loss_prop = np.average(valid_loss_prop)
            validation_losses.append(avg_validation_loss)
            validation_losses_pos.append(avg_valid_loss_pos)
            validation_losses_der.append(avg_valid_loss_der)
            validation_losses_prop.append(avg_valid_loss_prop)
            #print(f'Valid Loss: {avg_validation_loss:.10f}', end=' ')
            print('Valid Loss:',end=' ')
            print(f'Pos= {avg_valid_loss_pos:.10f}, Deriv= {avg_valid_loss_der:.14f}, Prop= {avg_valid_loss_prop:.10f}')
            log_file.write(f'Validation Epoch: {epoch} \t Loss: {avg_validation_loss:.10f}\n')
            
            # Update learning rate scheduler if available
            if scheduler_step is not None: 
                scheduler_step.step()
                old_lr = current_lr
                current_lr = optimizer.param_groups[0]['lr']
                if old_lr!=current_lr:
                    print(f'New learning rate: {current_lr}')
            if scheduler_reduce is not None: 
                scheduler_reduce.step(avg_validation_loss)
                old_lr = current_lr
                current_lr = optimizer.param_groups[0]['lr']
                if old_lr!=current_lr:
                    print(f'New learning rate: {current_lr}')
            torch.cuda.empty_cache()
            
            # Save model every 20 epochs
            if epoch % 20 == 0:
                #print(f"Train Losses: Pos: {avg_train_loss_pos}, Deriv:{avg_train_loss_der}, Prop:{avg_train_loss_prop}")
                #print(f"Valid Losses: Pos: {avg_valid_loss_pos}, Deriv:{avg_valid_loss_der}, Prop:{avg_valid_loss_prop}")
                epoch_path = os.path.join(model_dir, f"model_{epoch}.pt")
                torch.save(model, epoch_path)

    # Save the final model
    torch.save(model, model_path)
    
    # Plot and save loss graphs
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'{output_dir}/training_loss.png')
    plt.clf()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig(f'{output_dir}/test_train_loss.png')
    plt.show()
    plt.plot(validation_losses_pos, label='Valid prositive Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Positive Loss')
    plt.savefig(f'{output_dir}/pos_loss.png')
    plt.show()
    plt.plot(validation_losses_der, label='Valid derivative Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Derivative Loss')
    plt.savefig(f'{output_dir}/deriv_loss.png')
    plt.show()
    plt.plot(validation_losses_prop, label='Valid properness Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Properness Loss')
    plt.savefig(f'{output_dir}/prop_loss.png')
    plt.show()
'''

def train(data):  
    name = data.get("name", "main")
    check_alpha = data.get("check_alpha")
    model = data.get("model")
    epochs = data.get("epochs")
    batchsize = data.get("batchsize")
    optimizer = data.get("optimizer")
    scheduler_step = data.get("scheduler_step")
    scheduler_reduce = data.get("scheduler_reduce")
    ode_fn = data.get("ode_fn")
    domaindataset = data.get("domain_dataset")
    validationdomaindataset = data.get("validation_domain_dataset")
    if check_alpha:
        alpha = data.get("alpha")
        delta = data.get("delta")
        c = data.get("c")
        lie_params = [alpha, delta, c]
    else:
        lie_params = data.get("Q") 
    positive_w = data.get("positive_w")
    derivative_w = data.get("derivative_w")
    properness_w = data.get("properness_w")
    
    if scheduler_step!=None and scheduler_reduce!=None:
        raise ValueError('Both schedulers were given.')
        
    scheduler = None
    if scheduler_step!=None:
        scheduler = scheduler_step
    if scheduler_reduce!=None:
        scheduler = scheduler_reduce
    current_file = os.getcwd()
    dir_name = name + "_check_alpha" if check_alpha else name
    sys_dir = os.path.join(current_file, "output", dir_name)
    output_dir = os.path.join(sys_dir, dir_name+"_01")
    
    if os.path.exists(output_dir):
        counter = 2
        while True:
            i=str(counter)
            if counter<10:
                i = "0"+i
            output_dir = os.path.join(sys_dir, dir_name+"_"+i)
            if not os.path.exists(output_dir):
                break
            else:
                counter +=1
                
    model_dir = os.path.join(output_dir, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, f"model.pt")
    file_path = f"{output_dir}/train.txt"

    params_path = f"{output_dir}/params.json"
    params = {
        "name": name,
        "model": str(model),
        "epochs": epochs,
        "batchsize": batchsize,
        "optimizer": str(optimizer),
        "scheduler": str(scheduler.state_dict()) if scheduler != None else "None",
        "domainDataset": str(domaindataset),
        "validationDomainDataset": str(validationdomaindataset)
    } 
    with open(params_path, 'w') as fp:
        json.dump(params, fp)
    
    domain_dataloader = DataLoader(domaindataset, batch_size=batchsize, shuffle=True)
    validation_dataloader = DataLoader(validationdomaindataset, batch_size=batchsize, shuffle=False)

    train_losses = []  # To store training losses
    validation_losses = []  # To store validation losses
    validation_losses_pos = []
    validation_losses_der = []
    validation_losses_prop = []
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning rate: {current_lr}")
    with open(file_path, 'w') as log_file:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = []
            epoch_loss_pos = []
            epoch_loss_der = []
            epoch_loss_prop = []
            for batch_idx, x_in in enumerate(domain_dataloader):
                optimizer.zero_grad()
                x_in = torch.Tensor(x_in).to(torch.device('cuda:0'))
                loss_positive, loss_derivative, loss_properness = compute_loss(
                    name,\
                    check_alpha,\
                    x_in,\
                    model,\
                    ode_fn,\
                    lie_params,\
                    positive_w,\
                    derivative_w,\
                    properness_w
                )
                loss = loss_positive + loss_derivative + loss_properness
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                epoch_loss_pos.append(loss_positive.item())
                epoch_loss_der.append(loss_derivative.item())
                epoch_loss_prop.append(loss_properness.item())

            avg_train_loss = np.mean(epoch_loss)
            avg_train_loss_pos = np.mean(epoch_loss_pos)
            avg_train_loss_der = np.mean(epoch_loss_der)
            avg_train_loss_prop = np.mean(epoch_loss_prop)
            train_losses.append(avg_train_loss)
            print(f'Epoch {epoch} \t Train Loss: {avg_train_loss:.5f};', end= ' \t ')
            log_file.write(f'Epoch {epoch} \t Training Loss: {avg_train_loss:.10f}\n')

            # Validation and learning rate adjustment every 10 valids
            model.eval()
            validation_loss = []
            valid_loss_pos = []
            valid_loss_der = []
            valid_loss_prop = []
            for batch_idx, x_in in enumerate(validation_dataloader):
                x_in = torch.Tensor(x_in).to(torch.device('cuda:0'))
                val_loss_pos, val_loss_der, val_loss_prop = compute_loss(
                    name,\
                    check_alpha,\
                    x_in,\
                    model,\
                    ode_fn,\
                    lie_params,\
                    positive_w,\
                    derivative_w,\
                    properness_w
                )
                val_loss = val_loss_pos + val_loss_der + val_loss_prop
                validation_loss.append(val_loss.item())
                valid_loss_pos.append(val_loss_pos.item())
                valid_loss_der.append(val_loss_der.item())
                valid_loss_prop.append(val_loss_prop.item())
                
            avg_validation_loss = np.average(validation_loss)
            avg_valid_loss_pos = np.average(valid_loss_pos)
            avg_valid_loss_der = np.average(valid_loss_der)
            avg_valid_loss_prop = np.average(valid_loss_prop)
            validation_losses.append(avg_validation_loss)
            validation_losses_pos.append(avg_valid_loss_pos)
            validation_losses_der.append(avg_valid_loss_der)
            validation_losses_prop.append(avg_valid_loss_prop)
            #print(f'Valid Loss: {avg_validation_loss:.10f}', end=' ')
            print('Valid Loss:',end=' ')
            print(f'Pos= {avg_valid_loss_pos:.14f}, Deriv= {avg_valid_loss_der:.14f}, Prop= {avg_valid_loss_prop:.14f}')
            log_file.write(f'Validation Epoch: {epoch} \t Loss: {avg_validation_loss:.10f}\n')
            
            # Update learning rate scheduler if available
            if scheduler_step is not None: 
                scheduler_step.step()
                old_lr = current_lr
                current_lr = optimizer.param_groups[0]['lr']
                if old_lr!=current_lr:
                    print(f'New learning rate: {current_lr}')
            if scheduler_reduce is not None: 
                scheduler_reduce.step(avg_validation_loss)
                old_lr = current_lr
                current_lr = optimizer.param_groups[0]['lr']
                if old_lr!=current_lr:
                    print(f'New learning rate: {current_lr}')
            torch.cuda.empty_cache()
            
            # Save model every 20 epochs
            if epoch % 20 == 0:
                #print(f"Train Losses: Pos: {avg_train_loss_pos}, Deriv:{avg_train_loss_der}, Prop:{avg_train_loss_prop}")
                #print(f"Valid Losses: Pos: {avg_valid_loss_pos}, Deriv:{avg_valid_loss_der}, Prop:{avg_valid_loss_prop}")
                epoch_path = os.path.join(model_dir, f"model_{epoch}.pt")
                torch.save(model, epoch_path)

    # Save the final model
    torch.save(model, model_path)
    
    # Plot and save loss graphs
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'{output_dir}/training_loss.png')
    plt.clf()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig(f'{output_dir}/test_train_loss.png')
    plt.show()
    plt.plot(validation_losses_pos, label='Valid prositive Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Positive Loss')
    plt.savefig(f'{output_dir}/pos_loss.png')
    plt.show()
    plt.plot(validation_losses_der, label='Valid derivative Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Derivative Loss')
    plt.savefig(f'{output_dir}/deriv_loss.png')
    plt.show()
    plt.plot(validation_losses_prop, label='Valid properness Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Properness Loss')
    plt.savefig(f'{output_dir}/prop_loss.png')
    plt.show()

def train_param_opt(data):  
    name = data.get("name", "main")
    model = data.get("model")
    epochs = data.get("epochs")
    batchsize = data.get("batchsize")
    optimizer = data.get("optimizer")
    scheduler_step = data.get("scheduler_step")
    scheduler_reduce = data.get("scheduler_reduce")
    ode_fn = data.get("ode_fn")
    domaindataset = data.get("domain_dataset")
    validationdomaindataset = data.get("validation_domain_dataset")
    Q = data.get("Q") 
    positive_w = data.get("positive_w")
    derivative_w = data.get("derivative_w")
    properness_w = data.get("properness_w")
    normalization = data.get("normalization")
    if scheduler_step!=None and scheduler_reduce!=None:
        raise ValueError('Both schedulers were given.')
        
    scheduler = None
    if scheduler_step!=None:
        scheduler = scheduler_step
    if scheduler_reduce!=None:
        scheduler = scheduler_reduce
    current_file = os.getcwd()
    sys_dir = os.path.join(current_file, "output", name)
    output_dir = os.path.join(sys_dir, name+"_01")
    
    domain_dataloader = DataLoader(domaindataset, batch_size=batchsize, shuffle=True)
    validation_dataloader = DataLoader(validationdomaindataset, batch_size=batchsize, shuffle=False)

    train_losses = []  # To store training losses
    validation_losses = []  # To store validation losses
    current_lr = optimizer.param_groups[0]['lr']
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = []
        epoch_loss_pos = []
        epoch_loss_der = []
        epoch_loss_prop = []
        for batch_idx, x_in in enumerate(domain_dataloader):
            optimizer.zero_grad()
            x_in = torch.Tensor(x_in).to(torch.device('cuda:0'))
            loss_positive, loss_derivative, loss_properness = compute_loss(
                normalization,\
                x_in,\
                model,\
                ode_fn,\
                Q,\
                positive_w,\
                derivative_w,\
                properness_w
            )
            loss = loss_positive + loss_derivative + loss_properness
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            epoch_loss_pos.append(loss_positive.item())
            epoch_loss_der.append(loss_derivative.item())
            epoch_loss_prop.append(loss_properness.item())

        avg_train_loss = np.mean(epoch_loss)
        avg_train_loss_pos = np.mean(epoch_loss_pos)
        avg_train_loss_der = np.mean(epoch_loss_der)
        avg_train_loss_prop = np.mean(epoch_loss_prop)
        train_losses.append(avg_train_loss)

        # Validation and learning rate adjustment every 10 valids
        model.eval()
        validation_loss = []
        valid_loss_pos = []
        valid_loss_der = []
        valid_loss_prop = []
        for batch_idx, x_in in enumerate(validation_dataloader):
            x_in = torch.Tensor(x_in).to(torch.device('cuda:0'))
            val_loss_pos, val_loss_der, val_loss_prop = compute_loss(
                normalization,\
                x_in,\
                model,\
                ode_fn,\
                Q,\
                positive_w,\
                derivative_w,\
                properness_w
            )
            val_loss = val_loss_pos + val_loss_der + val_loss_prop
            validation_loss.append(val_loss.item())
            valid_loss_pos.append(val_loss_pos.item())
            valid_loss_der.append(val_loss_der.item())
            valid_loss_prop.append(val_loss_prop.item())
        avg_validation_loss = np.average(validation_loss)
        avg_valid_loss_pos = np.average(valid_loss_pos)
        avg_valid_loss_der = np.average(valid_loss_der)
        avg_valid_loss_prop = np.average(valid_loss_prop)
        validation_losses.append(avg_validation_loss)

        # Update learning rate scheduler if available
        if scheduler_step is not None: 
            scheduler_step.step()
            current_lr = optimizer.param_groups[0]['lr']            
        if scheduler_reduce is not None: 
            scheduler_reduce.step(avg_validation_loss)
            current_lr = optimizer.param_groups[0]['lr']
        torch.cuda.empty_cache()
    
    # Plot and save loss graphs
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.clf()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.show()
    return max(validation_loss)