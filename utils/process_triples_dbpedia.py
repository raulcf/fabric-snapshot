import pandas as pd


banned = ["22-rdf-syntax-ns#type", "rdf-schema#label", "rdf-schema#comment",
          "owl#sameAs", "subject", "wikiPageID", "wikiPageRevisionID", "wikiPageWikiLink",
          "wikiPageExternalLink", "prov#wasDerivedFrom", "vrank#hasRank", "vrank#rankValue",
          "depiction", "thumbnail", "caption", "isPrimaryTopicOf", "homepage", "imageSize",
          "website", "wordnet", "imagesize", "abstract"]

paths = [#"/Users/ra-mit/data/fabric/dbpedia/csv/jennifer_widom.csv",
         #"/Users/ra-mit/data/fabric/dbpedia/csv/hector_garcia_molina.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/michael_stonebraker.csv",
         #"/Users/ra-mit/data/fabric/dbpedia/csv/joe_hellerstein.csv",
         #"/Users/ra-mit/data/fabric/dbpedia/csv/jeffrey_naughton.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/dave_dewitt.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/sam_madden.csv",
         #"/Users/ra-mit/data/fabric/dbpedia/csv/stan_zdonik.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/hari_balakrishnan.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/butler_lampson.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/john_guttag.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/frans_kaashoek.csv",
         #"/Users/ra-mit/data/fabric/dbpedia/csv/mike_franklin.csv",
         #"/Users/ra-mit/data/fabric/dbpedia/csv/michael_jordan.csv",
         #"/Users/ra-mit/data/fabric/dbpedia/csv/david_blei.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/arvind.csv",
         #"/Users/ra-mit/data/fabric/dbpedia/csv/scott_aaronson.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/hal_abelson.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/bonnie_berger.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/erik_demaine.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/eric_grimson.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/alan_edelman.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/anant_agarwal.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/barbara_liskov.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/dina_katabi.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/joel_emer.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/shafi_goldwasser.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/piotr_indyk.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/daniel_jackson.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/leslie_kaelbling.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/david_karger.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/boris_katz.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/tom_leighton.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/charles_leiserson.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/john_leonard.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/andrew_lo.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/nancy_lynch.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/michel_goemans.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/thomas_magnanti.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/albert_meyer.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/silvio_micali.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/joel_moses.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/tomasso_poggio.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/ron_rivest.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/ronitt_rubinfeld.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/daniela_rus.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/jerry_saltzer.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/stephanie_seneff.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/julie_shah.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/nir_shavit.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/peter_shor.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/michael_sipser.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/gerald_sussman.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/joshua_tenenbaum.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/steve_ward.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/daniel_weitzner.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/ryan_williams.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/patrick_winston.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/jack_wisdom.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/victor_zue.csv",
         "/Users/ra-mit/data/fabric/dbpedia/csv/robert_morris.csv"]


o_path = "/Users/ra-mit/data/fabric/dbpedia/triples_structured/all.csv"

triples = []
total = 0
for p in paths:

    df = pd.read_csv(p)

    for idx, row in df.iterrows():
        total += 1
        s = row["subject"]
        p = row["predicate"]
        o = row["object"]
        s = s.split("/")[-1].replace(",", " ").strip()
        p = p.split("/")[-1].replace(",", " ").strip()
        o = o.split("/")[-1].replace(", ", " ").strip()
        triple = s + "," + p + "," + o
        if p not in banned:
            print(triple)
            triples.append(triple)

print("#generated triples for writing: " + str(len(triples)))
with open(o_path, "w") as f:
    f.write("s,p,o\n")
    for l in triples:
        f.write(l + '\n')

print("Total triples generated: " + str(total))

