


#ccg_tag_file = "data/orm_data/set0.tag"
#input_files = ["data/orm_data/set0-dev.interleave.eng"]
#output_files = ["data/orm_data/set0-dev.interleave.null.eng"]

ccg_tag_file = "data/kftt_data/kyoto.tag"
input_files = ["data/kftt_data/kyoto-train.interleave.en.filt", "data/kftt_data/kyoto-dev.interleave.en"]
output_files = ["data/kftt_data/kyoto-train.interleave.null.en.filt", "data/kftt_data/kyoto-dev.interleave.null.en"]


ccg_tags = []
with open(ccg_tag_file, 'r') as tag_file:
  for line in tag_file:
    ccg_tags.append(line.strip())
rep_tag = ccg_tags[0]
ccg_tags = set(ccg_tags)

for infile, outfile in zip(input_files, output_files):
  infile = open(infile, 'r', encoding='utf-8')
  outfile = open(outfile, 'w', encoding='utf-8')
  for line in infile:
    toks = line.split()
    for i in range(len(toks)):
      if toks[i] in ccg_tags:
        toks[i] = rep_tag
    outfile.write(' '.join(toks) + '\n')

infile.close()
outfile.close()
