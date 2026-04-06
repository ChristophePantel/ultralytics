# Christophe Pantel, IRIT, Toulouse Université

import torch
import torch.nn as nn

class VariantNotFound(Exception):
    pass

def get_class_names():
    class_names = frozenset({
        'Aeroplane', #00
       'Animal_Wing', #01
       'Animal', #02
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
        "Bird" : frozenset([ "Animal" ]),
        "Cat" : frozenset([ "Animal" ]),
        "Cow" : frozenset([ "Animal" ]),
        "Dog" : frozenset([ "Animal" ]),
        "Horse" : frozenset([ "Animal" ]),
        "Person" : frozenset([ "Animal" ]),
        "Sheep" : frozenset([ "Animal" ]),
        "Aeroplane" : frozenset([ "Vehicle" ]), 
        "Bicycle" : frozenset([ "Vehicle" ]),
        "Boat" : frozenset([ "Vehicle" ]),
        "Bus" : frozenset([ "Vehicle" ]),
        "Car" : frozenset([ "Vehicle" ]),
        "Motorbike" : frozenset([ "Vehicle" ]),
        "Train" : frozenset(["Vehicle"])
        }
    return refined_classes

def get_abstract_classes():
    abstract_classes = frozenset([
            "Animal",
            "Vehicle"
        ])
    return abstract_classes

def get_contained_classes():
    contained_classes = {
        "Animal" : frozenset([ "Eye", "Head", "Leg", "Neck", "Torso" ]),
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

def class_names_to_codes(class_names, class_name_to_code):
# names is a set of names
# name_to_code is the function that association a code to a name
# returns is a set of codes for each name
    class_codes = frozenset()
    for class_name in class_names:
        class_codes = class_codes.union( { class_name_to_code[class_name] })
    return class_codes

# Build the variant of an element according to the relation (paths in the relation whose root is the element)
def element_variants(element, relation):
    singleton_element = frozenset({element})
    if element in relation:
        result = frozenset( {} )
        for target in relation[element]:
            for variant in element_variants( target, relation):
                result = result.union( { variant.union( singleton_element ) } )
    else:
        result = frozenset( { singleton_element } )
    return result     

# Build the variants of all the elements according to the relation (paths in the relation whose roots are the elements)
def variants(elements,abstracts,relation):
    variant_to_element = {}
    code = 0
    result = {}
    all_variants = frozenset()
    for element in elements:
        if element not in abstracts:
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

def get_variant_code(variant_classes, variants):
    # variant_classes : set of classes in the variant
    # variants : dict which associate to a variant number, the set of classes
    # Return : number of the variant
    variant_number = 0
    for key, value in variants.items():
        if value == variant_classes:
            variant_number = key
        else: 
            raise VariantNotFound
    return variant_number
    
def get_candidate_variants(variant_classes, variant_to_classes):
    variant_set = frozenset()
    class_to_variants = invert_relation(variant_to_classes)
    for key, value in class_to_variants.items():
        if value==variant_classes:
            variant_set.intersection(key)
    return variant_set
    
# TODO : test A <-> B = A -> B /\ B -> A
def scores_fuzzy_equiv(batch_scores, prediction_scores, alpha=0.9, power=3):
    """Compute fuzzy equivalence between expected scores and predicted scores.
    
    Args:
    
    Returns:
    """
    batch_range = batch_scores.shape[0]
    prediction_range = prediction_scores.shape[0]
    aligned_batch_scores = torch.unsqueeze(batch_scores, 0).expand(prediction_range,-1,-1)
    aligned_prediction_scores = torch.unsqueeze(prediction_scores, 1).expand(-1,batch_range,-1)
    negated_aligned_batch_scores = 1.0 - aligned_batch_scores
    negated_aligned_prediction_scores = 1.0 - aligned_prediction_scores
    positive_component = aligned_batch_scores * aligned_prediction_scores
    negative_component = negated_aligned_batch_scores * negated_aligned_prediction_scores
    # Version Mean of positive contribution (bad according to CP/IRIT)
    # positive_component_sum = positive_component.sum(-1)
    # aligned_batch_scores_sum = aligned_batch_scores.sum(-1)
    # batch_prediction_equiv_mean = positive_component_sum  / aligned_batch_scores_sum 
    # Version Balanced Mean of positive and negative contribution
    batch_prediction_equiv = alpha *  positive_component + (1 - alpha) * negative_component
    # Version enhanced power mean
    # batch_prediction_equiv_mean = batch_prediction_equiv.pow(power).mean(-1).pow(1/power)
    # Version linear mean
    batch_prediction_equiv_mean = batch_prediction_equiv.mean(-1)
    return batch_prediction_equiv_mean

def scores_bce(batch_scores, prediction_scores):
    """Compute binary cross entropy between expected scores and predicted scores.
    
    Args:
    
    Returns:
    """
    bce_calculator = nn.BCELoss(reduction="none")
    batch_range = batch_scores.shape[0]
    prediction_range = prediction_scores.shape[0]
    batch_scores_sum = torch.sum(batch_scores,1)
    aligned_batch_scores_sum = torch.unsqueeze(batch_scores_sum, 0).expand(prediction_range,-1)
    aligned_batch_scores = torch.unsqueeze(batch_scores, 0).expand(prediction_range,-1,-1)
    aligned_prediction_scores = torch.unsqueeze(prediction_scores, 1).expand(-1,batch_range,-1)
    bce_per_class = bce_calculator(aligned_prediction_scores,aligned_batch_scores) 
    bce = torch.sum(bce_per_class,2) / batch_scores_sum
    # detected = torch.where(bce < 1)
    return bce
    
def generate_distance_matrix(class_number, class_variants, distance):
    # calcule la distance de chaque variant à chaque variant en utilisant la fuzzy logic ou la bce
    encoded_variants = encode_variants(class_number, class_variants)
    transpose_encoded_variants = torch.transpose(encoded_variants, 0, 1)
    result = distance(encoded_variants, transpose_encoded_variants)
    return result 

