# Christophe Pantel, IRIT, Toulouse Université

import torch

def get_class_names():
    class_names = frozenset({
        'Aeroplane', #00
       'Animal_Wing', #01
       'Animals', #02
       'Arm', #03
       'Artifact_Wing', #04
       'Beak', #05
       'Bicycle', #06
       'Bird', #07
       'Boat', #08
       'Body', #09
       'Bodywork', #10
       'Bottle', #11
       'Bus', #12
       'Cap', #13
       'Car', #14
       'Cat', #15
       'Chain_Wheel', #16
       'Chair', #17
       'Coach', #18
       'Cow', #19
       'Dining_Table', #20
       'Dog', #21
       'Door', #22
       'Ear', #23
       'Eyebrow', #24
       'Engine', #25
       'Eye', #26
       'Foot', #27
       'Hair', #28
       'Hand', #29
       'Handlebar', #30
       'Head', #31
       'Headlight', #32
       'Hoof', #33
       'Horn', #34
       'Horse', #35
       'Leg', #36
       'License_Plate', #37
       'Locomotive', #38
       'Mirror', #39
       'Motorbike', #40
       'Mouth', #41
       'Muzzle', #42
       'Neck', #43
       'Nose', #44
       'Person', #45
       'Plant', #46
       'Pot', #47
       'Potted_Plant', #48
       'Saddle', #49
       'Screen', #50
       'Sheep', #51
       'Sofa', #52
       'Stern', #53
       'Tail', #54
       'Torso', #55
       'Train', #56
       'TV_Monitor', #57
       'Vehicle', #58
       'Wheel', #59
       'Window', #60
    })
    return class_names

def get_refined_classes():
    refined_classes = {
        "Bird" : frozenset([ "Animals" ]),
        "Cat" : frozenset([ "Animals" ]),
        "Cow" : frozenset([ "Animals" ]),
        "Dog" : frozenset([ "Animals" ]),
        "Horse" : frozenset([ "Animals" ]),
        "Person" : frozenset([ "Animals" ]),
        "Sheep" : frozenset([ "Animals" ]),
        "Aeroplane" : frozenset([ "Vehicle" ]), 
        "Bicycle" : frozenset([ "Vehicle" ]),
        "Boat" : frozenset([ "Vehicle" ]),
        "Bus" : frozenset([ "Vehicle" ]),
        "Car" : frozenset([ "Vehicle" ]),
        "Motorbike" : frozenset([ "Vehicle" ]),
        "Train" : frozenset(["Vehicle"])
        }
    return refined_classes

def get_contained_classes():
    contained_classes = {
        "Animals" : frozenset([ "Eye", "Head", "Leg", "Neck", "Torso" ]),
        "Bird" : frozenset(["Animal_Wing", "Beak", "Tail"]),
        "Cat" : frozenset(["Ear", "Tail"]),
        "Cow" : frozenset(["Ear", "Horn", "Muzzle", "Tail"]),
        "Dog" : frozenset(["Ear", "Muzzle", "Nose", "Tail"]),
        "Horse" : frozenset(["Ear", "Hoof", "Muzzle", "Tail"]),
        "Person" : frozenset(["Arm", "Ear", "Eyebrow", "Foot", "Hair", "Hand", "Mouth", "Nose"]),
        "Sheep" : frozenset(["Ear", "Horn", "Muzzle", "Tail"]),
        "Bottle" : frozenset(["Body", "Cap"]),
        "Potted_Plant" : frozenset(["Plant", "Pot"]),
        "TV_Monitor" : frozenset(["Screen"]),
        "Aeroplane" : frozenset(["Artifact_Wing", "Body", "Engine", "Stern", "Wheel"]),
        "Bicycle" : frozenset(["Chain_Wheel", "Handlebar", "Headlight", "Saddle", "Wheel"]),
        "Bus" : frozenset(["Bodywork", "Door", "Headlight", "License_Plate", "Mirror", "Wheel", "Window"]),
        "Car" : frozenset(["Bodywork", "Door", "Headlight", "License_Plate", "Mirror", "Wheel", "Window"]),
        "Motorbike" : frozenset(["Handlebar", "Headlight", "Saddle", "Wheel"]),
        "Train" : frozenset([ "Coach", "Headlight", "Locomotive" ])
        }
    return contained_classes

def associate_number_to_class_and_class_to_number(classe_names):
    dictionnary_number_to_class = {}
    dictionnary_class_to_number = {}
    indice = 0
    for element in classe_names:
        dictionnary_number_to_class[indice] = element
        dictionnary_class_to_number[element] = indice
        indice += 1
    return dictionnary_number_to_class, dictionnary_class_to_number

def associate_codes_to_hierarchies(codes, named_hierarchy):
    coded_named_hierarchy = {}
    for key in named_hierarchy:
        values = named_hierarchy.get(key)
        coded_values = frozenset()
        for element in values:
            element_code = codes.get(element)
            coded_values = coded_values.union({element_code})
        coded_key = codes.get(key)
        coded_named_hierarchy[coded_key] = coded_values

    return coded_named_hierarchy

def invert_relation_v0(relation):
    inverted_relation = {}
    for key in relation:
        inverted_relation[key] = frozenset()
    print(inverted_relation)
    for key in relation:
        values = relation.get(key)
        for value in values:
            inverted_relation[value].add(key)

    return inverted_relation 

def followers(root,relation):
    result = frozenset({root})
    if root in relation:
        for target in relation[root]:
            result = result.union(followers(target, relation))
    return result     

def transitive_closure(relation):
    transitive_dict = {}
    for key in relation:
        transitive_dict[key] = frozenset({})
        for target in relation[key]:
            transitive_dict[key] = transitive_dict[key].union(followers(target, relation))
    return transitive_dict

def reflexive_transitive_closure(relation):
    reflexive_transitive_dict = {}
    for key in relation:
        reflexive_transitive_dict[key] = frozenset({key})
        for target in relation[key]:
            reflexive_transitive_dict[key] = reflexive_transitive_dict[key].union(followers(target, relation))
    return reflexive_transitive_dict

def roots(relation):
    values = frozenset()
    for key in relation:
        values = values.union(relation[key])
    results = frozenset()
    for key in relation:
        if key not in values:
            results = results.union({key})
    return results 

def leaves(relation):
    values = frozenset()
    for key in relation:
        values = values.union(relation[key])
    results = frozenset()
    for key in values:
        if key not in relation:
            results = results.union({key})
    return results 

def invert_relation(relation):
    inverted_relation = {}
    for key in relation:
        values = relation.get(key)
        for value in values:
            if value in inverted_relation:
                inverted_relation[value] = inverted_relation[value].union({key})
            else:
                inverted_relation[value] = frozenset({key})
    return inverted_relation

def element_variants(element, relation):
    singleton_element = frozenset({element})
    result = frozenset( { singleton_element } )
    if element in relation:
        for target in relation[element]:
            for variant in element_variants( target, relation):
                result = result.union( { variant.union( singleton_element ) } )
    return result     

def variants(elements,relation):
    variant_to_element = {}
    code = 0
    result = {}
    all_variants = frozenset()
    for element in elements:
        for variant in element_variants(element,relation):
            if variant not in all_variants:
                all_variants = all_variants.union({variant})
                result[code] = variant
                variant_to_element[code] = element
                code = code + 1
    return result, variant_to_element

def non_empty_keys(relation):
    non_empty_keys_results = list(relation.keys())
    for key in relation:
        if relation[key] == []:
            non_empty_keys_results.remove(key)
    return non_empty_keys_results

def expand(elements,relation):
    result = {}
    for element in elements:
        if element in relation:
            result[element] = relation[element]
        else:
            result[element] = frozenset()
    return result

def resolve(elements,composition,refinement):
    result = {}
    expanded_composition = expand(elements,composition)
    expanded_refinement = expand(elements,refinement)
    reflexive_transitive_refinement = reflexive_transitive_closure(expanded_refinement)
    for element in elements:
        result[element] = frozenset()
        for concept in reflexive_transitive_refinement[element]:
            result[element] = result[element].union(expanded_composition[concept])
    return result

def generalize(elements,relation,refinement):
    expanded_refinement = expand(elements,refinement)
    reflexive_transitive_refinement = reflexive_transitive_closure(expanded_refinement)
    result = {}
    for element in relation:
        result[element] = frozenset()
        for target in relation[element]:
            if target in reflexive_transitive_refinement:
                result[element] = result[element].union(reflexive_transitive_refinement[target]) 
    return result
            
def class_siblings(searched_class, ancestors):
    siblings = []
    inverted_ancestors = invert_relation(ancestors)
    parents_searched_class = ancestors.get(searched_class)
    for parent in parents_searched_class:
        children_search_class = inverted_ancestors.get(parent)
        for child in children_search_class:
            if child != searched_class:
                siblings.append(child)
    return siblings

def encode_variants(class_number, class_variants):
    result = torch.zeros((len(class_variants),class_number))
    for variant_index in class_variants:
        for class_index in class_variants[variant_index]:
            result[variant_index,class_index] = 1.0
    return result