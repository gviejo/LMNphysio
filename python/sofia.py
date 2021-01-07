#!/bin/python
# use this codon table (dictionary) to help
codonTable = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'
}

ReverseTable = {
    'A': 'T', 'T':'A', 'G':'C', 'C':'G'
}

def find_start_codon(seq):
    codons = [[], [], []]
    for i in range(len(seq)):
        if len(seq[i:i+3]) == 3:            
            codons[i%3].append(seq[i:i+3])

    all_start_end = []

    proteins = []

    for i, subcodons in enumerate(codons):
        start_codons = []
        for j, c in enumerate(subcodons):
            if c == 'ATG':
                start_codons.append(j)

        end_codons = []
        for j, c in enumerate(subcodons):
            if c in ['TAG', 'TAA', 'TGA']:
                end_codons.append(j)            

        start_end = []
        if len(start_codons) and len(end_codons):
            min_start = 0
            for k, start in enumerate(start_codons):
                if start > min_start:
                    for l, end in enumerate(end_codons):
                        if end > start:
                            start_end.append([start, end])
                            break
                min_start = end                
        
        if len(start_end):
            for i, s in enumerate(start_end):
                prot = "".join([codonTable[c] for c in subcodons[s[0]:s[1]]])
                proteins.append(prot)


    if len(proteins):
        return proteins        
    else:
        return 0

    

dnaSeq = 'AGCCATGTAGCTAACTCAGGTTACATGGGGATGACCCCGCGACTTGGATTAGAGTCTCTTTTGGAATAAGCCTGAATGATCCGAGTAGCATCTCAG'

dnaSeq = dnaSeq.upper() #make all letters capital letters

#Make reverse sequence
dnatranslate = ""
for letter in dnaSeq:
    dnatranslate += ReverseTable[letter]
reversedna = dnatranslate[::-1] #Reverse strand is read backwards


proteins_sequences = find_start_codon(dnaSeq)

print(len(proteins_sequences))

max_length = 1
idx = 0
for i, c in enumerate(proteins_sequences):
    if len(c) > max_length: 
        max_length = len(c)
        idx = i
        

print(proteins_sequences[idx])