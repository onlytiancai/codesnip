kubectl taint nodes --all node-role.kubernetes.io/master-
kubectl create -f notebook.yaml
kubectl get po
kubectl describe pod notebook-pod
curl -L 10.244.0.4:8888
kubectl expose pod notebook-pod --type=NodePort
kubectl get service notebook-pod

systemctl edit docker
    ExecStartPost=/sbin/iptables -I FORWARD -s 0.0.0.0/0 -j ACCEPT
systemctl daemon-reload
systemctl restart docker

kubectl create -f CreateServiceAccount.yaml
kubectl create -f RoleBinding.yaml

# get token
kubectl describe secret $(kubectl get secret -n kube-system | grep ^admin-user | awk '{print $1}') -n kube-system | grep -E '^token'| awk '{print $2}'
