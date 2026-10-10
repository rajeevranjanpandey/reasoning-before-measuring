"""Local SLM policies with a process deadline and complete action records."""
import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
import re
import resource
import subprocess
import time
import traceback
import numpy as np
import joblib
from experiment import ROOT, SEED, load, state_view

MODEL_FILES={'qwen':'Qwen3-1.7B-Q8_0.gguf','granite':'granite-3.3-2b-instruct-Q8_0.gguf'}

def worker(conn,model,gpu):
    try:
        from llama_cpp import Llama, LlamaGrammar, __version__
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        t=time.perf_counter()
        llm=Llama(model_path=str(ROOT/'models'/MODEL_FILES[model]),n_ctx=4096,
                  n_threads=4,n_gpu_layers=-1 if gpu else 0,verbose=False,seed=17)
        template=llm.metadata['tokenizer.chat_template']
        (ROOT/'provenance'/f'{model}_chat_template.jinja').write_text(template)
        env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
        env.filters['tojson']=lambda x,**kw:json.dumps(x,**kw)
        def fail(x): raise ValueError(x)
        env.globals['raise_exception']=fail
        env.globals['strftime_now']=lambda fmt:datetime.datetime(2026,9,20).strftime(fmt)
        tmpl=env.from_string(template)
        eos=llm.detokenize([llm.token_eos()],special=True).decode()
        bos=llm.detokenize([llm.token_bos()],special=True).decode()
        conn.send({'ready':True,'load_seconds':time.perf_counter()-t,'runtime':__version__,
                   'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        while True:
            req=conn.recv()
            if req is None:break
            prompt=tmpl.render(messages=req['messages'],bos_token=bos,eos_token=eos,
                               add_generation_prompt=True,enable_thinking=False,thinking=False,tools=None)
            # llama.cpp retains only identical prompt-prefix tokens and removes
            # the previous differing suffix. No previous episode messages enter.
            llm.set_seed(req['seed'])
            action_grammar=LlamaGrammar.from_json_schema(json.dumps({'type':'object','properties':{
                'sensor':{'type':'integer','enum':req.get('legal_sensors',list(range(1,42)))}},
                'required':['sensor'],'additionalProperties':False}),verbose=False)
            plan_grammar=LlamaGrammar.from_json_schema(json.dumps({'type':'object','properties':{
                'sensors':{'type':'array','items':{'type':'integer','enum':list(range(1,42))},
                           'minItems':5,'maxItems':5}},'required':['sensors'],
                           'additionalProperties':False}),verbose=False)
            t=time.perf_counter()
            out=llm.create_completion(prompt,max_tokens=req.get('max_tokens',32),
                                      temperature=0.7,top_p=0.8,top_k=20,stop=[eos],
                                      seed=req['seed'],
                                      grammar=action_grammar if req.get('max_tokens',32)==32 else plan_grammar)
            conn.send({'text':out['choices'][0]['text'],'usage':out['usage'],
                       'finish_reason':out['choices'][0]['finish_reason'],
                       'generation_seconds':time.perf_counter()-t,'prompt':prompt,
                       'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        llm.close()
    except BaseException:
        conn.send({'error':traceback.format_exc()})
    finally:conn.close()

class LocalModel:
    def __init__(self,model,gpu=True):
        context=mp.get_context('spawn');self.conn,child=context.Pipe()
        self.process=context.Process(target=worker,args=(child,model,gpu))
        self.process.start()
        if not self.conn.poll(120):
            self.close();raise TimeoutError('Model load timeout')
        self.info=self.conn.recv()
        if not self.info.get('ready'):raise RuntimeError(self.info)
    def call(self,messages,seed,max_tokens=32,legal_sensors=None):
        t=time.perf_counter()
        try:self.conn.send(dict(messages=messages,seed=seed,max_tokens=max_tokens,
                               legal_sensors=legal_sensors or list(range(1,42))))
        except (BrokenPipeError,EOFError,OSError) as exc:
            return {'error':str(exc),'seconds':time.perf_counter()-t}
        if not self.conn.poll(90):
            self.close();return {'error':'90-second hard deadline','seconds':time.perf_counter()-t}
        try:result=self.conn.recv()
        except (EOFError,OSError) as exc:result={'error':str(exc)}
        result['seconds']=time.perf_counter()-t;return result
    def close(self):
        if self.process.is_alive():self.process.terminate()
        self.process.join(5)

def parse_action(text,legal):
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S).strip()
    text=re.sub(r'^```(?:json)?\s*|\s*```$','',text).strip()
    try:
        obj=json.loads(text)
        if set(obj)=={'sensor'} and type(obj['sensor']) is int and obj['sensor']-1 in legal:
            return obj['sensor']-1
    except (ValueError,TypeError):pass
    return None

def system_prompt(metadata):
    sensors='\n'.join(f'{j+1}: {v}' for j,v in metadata.items())
    return ('Select one sensor to reveal for Tennessee Eastman fault diagnosis. '
            'All unobserved values are unavailable. The diagnosis probabilities are for classes 0 through 20. '
            'Choose only a legal sensor. Return exactly {"sensor":integer}, no explanation. '
            'Sensor dictionary:\n'+sensors)

def run(model,stage,gpu=True):
    package=joblib.load(ROOT/'results/training.joblib');p=package['predictor']
    fallback=package['rankings']['mi'];system=system_prompt(package['metadata'])
    if stage=='development':
        x,y,r=load('Training');keep=(r>=401)&(r<=420)
        ix=np.flatnonzero(keep);ix=np.random.default_rng(SEED).choice(ix,20,replace=False)
        x,y,r=x[ix],y[ix],r[ix];seeds=[17]
    else:
        x,y,r=load('Testing');keep=(r>=1)&(r<=20);x,y,r=x[keep],y[keep],r[keep];seeds=[17,29,43]
    endpoint=LocalModel(model,gpu)
    (ROOT/'results'/f'{model}_{stage}_runtime.json').write_text(json.dumps(endpoint.info,indent=2))
    output=ROOT/'results'/f'{model}_{stage}.jsonl'
    completed=set()
    if output.exists():
        for line in output.read_text().splitlines():
            prior=json.loads(line)
            key=(prior['policy'],prior['seed'],prior['label'],prior['run'])
            if key in completed:raise RuntimeError('Duplicate outcome key')
            completed.add(key)
    try:
        # Warm-up excluded from episode timing, retained for provenance.
        warm=endpoint.call([{'role':'user','content':'Return exactly {"sensor":1}.'}],17)
        (ROOT/'results'/f'{model}_{stage}_warmup.json').write_text(json.dumps(warm,indent=2))
        if 'error' in warm:raise RuntimeError('Warm-up failed: '+warm['error'])
        with output.open('a') as f:
            for variant in ['raw','assisted']:
                for seed in seeds:
                    for index,(row,label,run_id) in enumerate(zip(x,y,r)):
                        if (variant,seed,int(label),int(run_id)) in completed:continue
                        start=time.perf_counter();obs={};actions=[];fallback_count=0
                        for step in range(3):
                            legal=[j for j in range(41) if j not in obs]
                            tool_start=time.perf_counter()
                            scores=p.scores(obs,legal) if variant=='assisted' else None
                            tool_seconds=time.perf_counter()-tool_start
                            state=state_view(obs,legal,3-step,p,package['metadata'],scores)
                            messages=[{'role':'system','content':system},
                                      {'role':'user','content':json.dumps(state,separators=(',',':'))}]
                            attempts=[];j=None
                            for attempt in range(2):
                                result=endpoint.call(messages,seed+step*101+attempt*1009,
                                                     legal_sensors=[j+1 for j in legal])
                                attempts.append(result)
                                j=parse_action(result.get('text',''),legal)
                                if j is not None:break
                                if 'error' in result:
                                    endpoint.close()
                                    endpoint=LocalModel(model,gpu)
                                messages=messages+[{'role':'assistant','content':result.get('text','')},
                                    {'role':'user','content':'Invalid response. Return exactly one JSON object with key sensor and a legal integer.'}]
                            assisted=j is None
                            if assisted:j=next(j for j in fallback if j in legal);fallback_count+=1
                            obs[j]=float(row[j])
                            actions.append(dict(sensor=j+1,value=obs[j],state=state,attempts=attempts,
                                                fallback=assisted,tool_seconds=tool_seconds))
                        prob=p.predict(obs)
                        record=dict(model=model,policy=variant,stage=stage,seed=seed,label=int(label),
                            run=int(run_id),budget=3,prediction=int(prob.argmax()),correct=int(prob.argmax()==label),
                            seconds=time.perf_counter()-start,fallback_count=fallback_count,actions=actions,
                            log_loss=float(-np.log(max(prob[label],1e-12))))
                        f.write(json.dumps(record)+'\n');f.flush()
                        completed.add((variant,seed,int(label),int(run_id)))
                        (ROOT/'results'/f'{model}_{stage}_progress.json').write_text(json.dumps({
                            'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            'completed_episodes':len(completed),'expected_episodes':40 if stage=='development' else 2520,
                            'variant':variant,'seed':seed,'latest_episode_seconds':record['seconds']},indent=2))
                        print(model,stage,variant,seed,index+1,'seconds',round(record['seconds'],2),
                              'fallbacks',fallback_count,flush=True)
    finally:endpoint.close()

def fixed_plan(model,gpu=True):
    package=joblib.load(ROOT/'results/training.joblib')
    endpoint=LocalModel(model,gpu)
    system=system_prompt(package['metadata']).replace('Return exactly {"sensor":integer}, no explanation.',
        'Return exactly {"sensors":[a,b,c,d,e]} with five distinct integer sensor IDs in acquisition order, no explanation.')
    messages=[{'role':'system','content':system},{'role':'user','content':json.dumps({
        'task':'Choose one reusable five-sensor acquisition plan before any test episodes.',
        'legal_sensors':list(range(1,42)),
        'diagnosis_probabilities':np.round(package['predictor'].predict({}),5).tolist()})}]
    attempts=[];ranking=None
    try:
        for attempt in range(2):
            out=endpoint.call(messages,17+attempt*1009,max_tokens=128);attempts.append(out)
            text=re.sub(r'<think>.*?</think>','',out.get('text',''),flags=re.S).strip()
            text=re.sub(r'^```(?:json)?\s*|\s*```$','',text).strip()
            try:
                obj=json.loads(text);candidate=obj['sensors']
                if len(candidate)==5 and len(set(candidate))==5 and all(type(j)is int and 1<=j<=41 for j in candidate):
                    ranking=[j-1 for j in candidate];break
            except (ValueError,TypeError,KeyError):pass
            messages+=[{'role':'assistant','content':out.get('text','')},
                       {'role':'user','content':'Return only valid JSON with five different legal IDs under the key sensors.'}]
        result=dict(model=model,ranking=ranking,attempts=attempts,runtime=endpoint.info,
                    setup_seconds=sum(a['seconds'] for a in attempts))
        (ROOT/'results'/f'{model}_fixed_plan.json').write_text(json.dumps(result,indent=2))
        if ranking is None:raise RuntimeError('Fixed plan did not produce five legal sensors')
        x,y,r=load('Testing');keep=r<=20;x,y,r=x[keep],y[keep],r[keep]
        with (ROOT/'results'/f'{model}_fixed_Testing_1_20.jsonl').open('w') as f:
            for row,label,run_id in zip(x,y,r):
                for budget in (1,3,5):
                    start=time.perf_counter();obs={j:float(row[j]) for j in ranking[:budget]}
                    prob=package['predictor'].predict(obs)
                    f.write(json.dumps(dict(model=model,policy='fixed_plan',seed=0,label=int(label),run=int(run_id),
                        budget=budget,prediction=int(prob.argmax()),correct=int(prob.argmax()==label),
                        seconds=time.perf_counter()-start,trajectory=[j+1 for j in ranking[:budget]]))+'\n')
        print('fixed plan',model,ranking,flush=True)
    finally:endpoint.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=MODEL_FILES)
    parser.add_argument('stage',choices=['development','pilot','plan']);parser.add_argument('--cpu',action='store_true')
    args=parser.parse_args()
    if args.stage=='plan':fixed_plan(args.model,not args.cpu)
    else:run(args.model,args.stage,not args.cpu)
