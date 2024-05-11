# This function takes a DNA sequence and a list containing the U2-type introns
# and returns a list containing all the U2-type introns within that DNA sequence.

def find_intron_in_sequence(sequence, intron_list):
    
    lst =  [intron.strip() for intron in intron_list if intron.strip() in sequence]
    return list(dict.fromkeys(lst))

# This function takes a DNA sequence and a list containing the U2-type introns
# and return True if the DNA sequence contains more than one intron, otherwise False.

def contains_multiple_introns(sequence, intron_list):
    count = 0
    for intron in intron_list:
        if intron in sequence:
            count += 1
            if count == 2:
                return True
    return False
    
