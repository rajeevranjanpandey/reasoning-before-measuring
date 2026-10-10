"""Post-pilot control: grammar permits specifically requested high/low sensor IDs."""
import json
from agents import LocalModel, parse_action
from experiment import ROOT

def main():
    assert json.loads((ROOT/'results/analysis.json').read_text())['primary_pilot_complete']
    records=[]
    for model in ('qwen','granite'):
        endpoint=LocalModel(model,True)
        try:
            for sensor in (7,23,41):
                messages=[{'role':'user','content':f'Return exactly {{"sensor":{sensor}}}. Select sensor {sensor}; do not choose a different sensor.'}]
                result=endpoint.call(messages,17,legal_sensors=list(range(1,42)))
                selected=parse_action(result.get('text',''),list(range(41)))
                records.append({'model':model,'requested':sensor,'selected':None if selected is None else selected+1,
                                'passed':selected==sensor-1,'result':result,'runtime':endpoint.info})
        finally:endpoint.close()
    output={'status':'post-pilot descriptive interface control; not a diagnosis trial','records':records,
            'passed':sum(r['passed'] for r in records),'total':len(records)}
    (ROOT/'results/interface_control.json').write_text(json.dumps(output,indent=2))
    print(json.dumps({'interface_control_passed':output['passed'],'total':output['total']}))

if __name__=='__main__':main()
