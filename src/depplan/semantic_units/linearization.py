"""Linearize AMR relations into semantic units with one-hop context.

Core arguments retain their predicate frame and ARG index.  A frame-specific
role is identified by that pair; this module does not invent a natural-language
PropBank role gloss or treat ARG0/ARG1/ARG2 as universal agent/object roles.
"""
import json
import re
from collections import Counter
from pathlib import Path

TEMPLATE_PATH = Path(__file__).with_name('amr_templates.json')
CORE_ARGUMENT = re.compile(r':(ARG\d+)(-of)?')


def collect_var2concept_and_names(triples):
    concepts = {s:t for s,r,t in triples if r==':instance'}
    name_nodes = {t for _,r,t in triples if r==':name'}
    tokens = {}
    for s,r,t in triples:
        if s in name_nodes and re.fullmatch(r':op\d+',r):
            try: value=json.loads(t)
            except (ValueError,TypeError): value=str(t).strip('"')
            tokens.setdefault(s,[]).append((int(r[3:]),str(value)))
    names = {v:' '.join(t for _,t in sorted(parts)) for v,parts in tokens.items()}
    owners = {s:names[t] for s,r,t in triples if r==':name' and t in names}
    return concepts,names,owners


def build_semantic_units(triples, template_path=TEMPLATE_PATH):
    triples = [tuple(map(str,t)) for t in triples]
    concepts,names,owners = collect_var2concept_and_names(triples)
    if not concepts:
        raise ValueError('AMR has no concept nodes')
    templates = json.loads(Path(template_path).read_text())
    # AMR (book :poss person) means the person possesses the book.
    templates[':poss'] = '{H} is possessed by {T}.'
    duplicates = Counter(concepts.values())
    relations = [t for t in triples if t[1]!=':instance']

    def surface(value):
        if value in owners:
            return owners[value]
        if value in names:
            return 'name '+names[value]
        if value in concepts:
            label = concepts[value]
            if label=='amr-unknown':label='unknown value'
            return f'{label} ({value})' if duplicates[concepts[value]]>1 else label
        try: decoded=json.loads(value)
        except (ValueError,TypeError):decoded=value
        return str(decoded)

    def render(triple):
        s,r,t = triple
        h,target = surface(s),surface(t)
        argument = CORE_ARGUMENT.fullmatch(r)
        if argument:
            role, inverse = argument.groups()
            predicate_node, filler_node = (t,s) if inverse else (s,t)
            # Inverse edges reverse the graph notation, not the frame's role.
            # Use the actual concept label even if the predicate has a :name.
            predicate = concepts.get(predicate_node, predicate_node)
            if predicate_node in concepts and duplicates[predicate]>1:
                predicate += f' ({predicate_node})'
            template = templates.get(':'+role, '{T} fills the {R} role of frame {P}.')
            return template.format(H=predicate,T=surface(filler_node),P=predicate,R=role)
        if r==':name' and s in owners:
            head = concepts.get(s,s)
            if duplicates[concepts.get(s,s)]>1:head+=f' ({s})'
            return f'The name of {head} is {owners[s]}.'
        if re.fullmatch(r':op\d+',r):
            return f'{target} is member {int(r[3:])} of {h}.'
        if r==':polarity' and t=='-':
            return f'It is not the case that {h}.'
        template = templates.get(r)
        if template:
            return template.format(H=h,T=target,P=h,R=r)
        return f'{h} has relation {r} to {target}.'

    instantiated = []
    units = []
    for i,(s,r,t) in enumerate(relations):
        # Variable IDs are retained for context matching: equal concepts are not equal nodes.
        instantiated.append({'source':s,'role':r,'target':t,
                             'source_concept':concepts.get(s,s),'target_concept':concepts.get(t,t)})
        variables = {v for v in (s,t) if v in concepts}
        context_indices = [j for j,(a,_,b) in enumerate(relations)
                           if j!=i and variables&{v for v in (a,b) if v in concepts}]
        center = render((s,r,t))
        context_texts = list(dict.fromkeys(render(relations[j]) for j in context_indices))
        text = center + (' Context: '+' '.join(context_texts) if context_texts else '')
        units.append({'id':i+1,'text':text,'central_text':center,'triple':[s,r,t],
                      'context_triples':[list(relations[j]) for j in context_indices],
                      'context_texts':context_texts})
    return {'semantic_units':[u['text'] for u in units],'unit_details':units,
            'instantiated_triples':instantiated,'variable_concepts':concepts,'entity_names':owners,
            'linearization':'frame-indexed-roles-with-variable-aware-one-hop-context-v2'}
