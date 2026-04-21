import json, numpy as np, math, time, pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple
from scipy.sparse.csgraph import minimum_spanning_tree as _mst_scipy
from scipy.sparse import csr_matrix as _csr

@dataclass
class City:
    node_id:int; x:float; y:float; ready_time:float; due_date:float; service_time:float

@dataclass
class TSPTWInstance:
    cities:List[City]; dist:np.ndarray=field(repr=False)
    @property
    def n(self): return len(self.cities)

def load_json(fp):
    with open(fp, encoding='utf-8') as f: data=json.load(f)
    scale=data['meta']['scale']; nodes=[data['depot']]+data['clients']
    cities=[City(nd['id'],nd['x']*scale,nd['y']*scale,nd.get('a',0) or 0,
                 nd.get('b',data['meta']['horizon']) or data['meta']['horizon'],nd['service']) for nd in nodes]
    coords=np.array([[c.x,c.y] for c in cities])
    dist=np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=-1)
    inst=TSPTWInstance(cities,dist); pm=defaultdict(list)
    for p in data.get('perturbations',[]):
        key=(min(p['arc'][0],p['arc'][1]),max(p['arc'][0],p['arc'][1])); pm[key].append((p['t_start'],p['t_end'],p['alpha']))
    inst._perturb_map=pm; return inst

def edge_cost_dyn(i,j,t,inst):
    base=inst.dist[i,j]; key=(min(i,j),max(i,j))
    for t0,t1,a in inst._perturb_map.get(key,[]):
        if t0<=t<=t1: return base*a
    return base

def tour_penalized_cost(tour,inst,penalty=200.0):
    t=0.0; cost=0.0; viol=0.0
    for k in range(len(tour)):
        city=inst.cities[tour[k]]
        if k>0:
            d=edge_cost_dyn(tour[k-1],tour[k],t,inst); cost+=d; t+=d
        t=max(t,city.ready_time); viol+=max(0.0,t-city.due_date); t+=city.service_time
    cost+=edge_cost_dyn(tour[-1],tour[0],t,inst)
    return cost+penalty*viol

def tour_cost_dynamic(tour,inst):
    t=0.0; cost=0.0
    for k in range(len(tour)-1):
        c=edge_cost_dyn(tour[k],tour[k+1],t,inst); cost+=c; t+=c
        t=max(t,inst.cities[tour[k+1]].ready_time); t+=inst.cities[tour[k+1]].service_time
    cost+=edge_cost_dyn(tour[-1],tour[0],t,inst)
    return cost

def tw_violation(tour,inst):
    t=0.0; total=0.0
    for k in range(len(tour)):
        city=inst.cities[tour[k]]
        if k>0: t+=inst.dist[tour[k-1],tour[k]]
        t=max(t,city.ready_time); total+=max(0.0,t-city.due_date); t+=city.service_time
    return total

def is_feasible(tour,inst):
    t=0.0
    for k in range(len(tour)):
        city=inst.cities[tour[k]]
        if k>0: t+=inst.dist[tour[k-1],tour[k]]
        t=max(t,city.ready_time)
        if t>city.due_date: return False
        t+=city.service_time
    return True

def one_tree_lb(inst):
    sub=_csr(inst.dist[1:,1:])
    lb=float(_mst_scipy(sub).sum())
    depot_edges=np.sort(inst.dist[0,1:])
    return lb+depot_edges[0]+depot_edges[1]

def earliest_feasible_tw(inst):
    n=inst.n; clients=sorted(range(1,n),key=lambda i:inst.cities[i].ready_time); tour=[0]
    for city in clients:
        best_pos=None; best_cost=float('inf')
        for pos in range(1,len(tour)+1):
            candidate=tour[:pos]+[city]+tour[pos:]
            if is_feasible(candidate,inst):
                c=sum(inst.dist[candidate[k],candidate[k+1]] for k in range(len(candidate)-1))+inst.dist[candidate[-1],candidate[0]]
                if c<best_cost: best_cost=c; best_pos=pos
        if best_pos is None: tour.append(city)
        else: tour.insert(best_pos,city)
    return tour

def get_neighbor_positions(part_positions,tour,inst,p):
    part_set=set(part_positions); part_cities=[tour[pos] for pos in part_positions]
    scores=[]
    for pos in range(len(tour)):
        if pos in part_set: continue
        city_idx=tour[pos]; min_dist=min(inst.dist[city_idx,c] for c in part_cities)
        scores.append((min_dist,pos))
    scores.sort(); return [pos for _,pos in scores[:p]]

def two_opt_subproblem(tour,sub_indices,inst):
    improved=False; best_tour=tour[:]; best_cost=tour_penalized_cost(best_tour,inst); n_sub=len(sub_indices)
    for a in range(n_sub-1):
        for b in range(a+2,n_sub):
            sub_cities=[best_tour[p] for p in sub_indices]
            new_sub=sub_cities[:a+1]+sub_cities[a+1:b+1][::-1]+sub_cities[b+1:]
            candidate=best_tour[:]
            for idx,pos in enumerate(sub_indices): candidate[pos]=new_sub[idx]
            c=tour_penalized_cost(candidate,inst)
            if c<best_cost-1e-9: best_cost=c; best_tour=candidate; improved=True
    return best_tour,improved

def or_opt_subproblem(tour,sub_indices,inst,segment_sizes=(1,2,3)):
    improved=False; best_tour=tour[:]; best_cost=tour_penalized_cost(best_tour,inst); n_sub=len(sub_indices)
    for seg_size in segment_sizes:
        for start in range(n_sub-seg_size):
            sub_cities=[best_tour[p] for p in sub_indices]
            segment=sub_cities[start:start+seg_size]; remaining=sub_cities[:start]+sub_cities[start+seg_size:]
            best_local=best_cost; best_ins=None
            for ins_pos in range(len(remaining)+1):
                if ins_pos==start: continue
                new_sub=remaining[:ins_pos]+segment+remaining[ins_pos:]
                candidate=best_tour[:]
                for idx,pos in enumerate(sub_indices): candidate[pos]=new_sub[idx]
                c=tour_penalized_cost(candidate,inst)
                if c<best_local-1e-9: best_local=c; best_ins=ins_pos
            if best_ins is not None:
                sub_cities=[best_tour[p] for p in sub_indices]
                segment=sub_cities[start:start+seg_size]; remaining=sub_cities[:start]+sub_cities[start+seg_size:]
                new_sub=remaining[:best_ins]+segment+remaining[best_ins:]
                for idx,pos in enumerate(sub_indices): best_tour[pos]=new_sub[idx]
                best_cost=best_local; improved=True
    return best_tour,improved

def popmusic(inst,r=10,p=20,max_iter=10,local_search='2opt'):
    n=inst.n; stats={'iter_costs':[],'iter_times':[],'improvements_per_iter':[],'feasible':None}
    tour=earliest_feasible_tw(inst)
    for iteration in range(max_iter):
        t0=time.perf_counter(); global_improved=False; iter_imp=0
        parts=[list(range(start,min(start+r,n))) for start in range(1,n,r)]
        for part_positions in parts:
            if len(part_positions)<2: continue
            neighbor_positions=get_neighbor_positions(part_positions,tour,inst,p)
            sub_positions=sorted((set(part_positions)|set(neighbor_positions))-{0})
            if local_search=='2opt': new_tour,improved=two_opt_subproblem(tour,sub_positions,inst)
            else: new_tour,improved=or_opt_subproblem(tour,sub_positions,inst)
            if improved:
                # Verify tour validity
                if len(new_tour)==n and len(set(new_tour))==n:
                    tour=new_tour; global_improved=True; iter_imp+=1
        elapsed=time.perf_counter()-t0
        stats['iter_costs'].append(tour_cost_dynamic(tour,inst))
        stats['iter_times'].append(elapsed)
        stats['improvements_per_iter'].append(iter_imp)
        if not global_improved: break
    stats['feasible']=is_feasible(tour,inst)
    return tour,stats

DATASET_DIR=pathlib.Path('./datasets')
dataset_files=sorted(DATASET_DIR.glob('tsptwd_n*.json'),key=lambda f:int(f.stem.split('_n')[1]))
dataset_files=[f for f in dataset_files if int(f.stem.split('_n')[1])<=1000]
print('Datasets:',[f.name for f in dataset_files])

results=[]
for fpath in dataset_files:
    n_nodes=int(fpath.stem.split('_n')[1])
    print('-'*55)
    print(f'  {fpath.name}  (n={n_nodes})')
    inst=load_json(str(fpath))
    r=max(5,min(25,n_nodes//20)); p=2*r
    ls='oropt' if n_nodes<=200 else '2opt'
    max_it=max(5,min(15,300//r))
    lb=one_tree_lb(inst)

    t0=time.perf_counter()
    tour,stats=popmusic(inst,r=r,p=p,max_iter=max_it,local_search=ls)
    t_exec=time.perf_counter()-t0

    cost_dyn=tour_cost_dynamic(tour,inst)
    violation=tw_violation(tour,inst)
    feasible=violation<1e-3
    ratio=cost_dyn/lb if lb>0 else float('nan')

    print(f'  1-tree LB      : {round(lb,2)}')
    print(f'  Cout dyn       : {round(cost_dyn,2)}  (distances + perturbations)')
    print(f'  Violation TW   : {round(violation,2)} min  ({"FAISABLE" if feasible else "INFAISABLE"})')
    print(f'  Ratio dyn/LB   : {round(ratio,4)}')
    print(f'  Temps exec     : {round(t_exec,3)}s')
    print(f'  Iterations     : {len(stats["iter_costs"])}')
    print(f'  Tour valide    : len={len(tour)} unique={len(set(tour))}')

    results.append({'n':n_nodes,'1tree_lb':round(lb,3),'cout_dyn':round(cost_dyn,3),
                    'violation_tw_min':round(violation,2),'faisable':feasible,
                    'ratio_dyn_lb':round(ratio,4),'temps_s':round(t_exec,3),
                    'iterations':len(stats['iter_costs'])})

print('='*55)
print(f"{'n':>6}  {'1tree_lb':>9}  {'cout_dyn':>9}  {'violation_tw':>12}  {'faisable':>8}  {'ratio':>7}  {'temps_s':>8}  {'iter':>4}")
for r in results:
    print(f"{r['n']:>6}  {r['1tree_lb']:>9.1f}  {r['cout_dyn']:>9.1f}  {r['violation_tw_min']:>12.1f}  {str(r['faisable']):>8}  {r['ratio_dyn_lb']:>7.4f}  {r['temps_s']:>8.3f}  {r['iterations']:>4}")
