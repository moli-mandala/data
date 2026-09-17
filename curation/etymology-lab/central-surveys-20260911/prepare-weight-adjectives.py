from research_helpers import Batch
b=Batch(29)
for l,words in [('Malvi','bajini|vajani'),('Nimadi','bajni'),('Bagheli','beḍženi|veḍženi')]:
 b.add(l,'heavy',words,'f_7u6j2wcwpjlti','Platts explicitly gives waznī and colloquial wazanī as adjectives meaning weighty/heavy; both the full entry and its scanned page were checked. The existing Hindi wazanī supplies the immediate donor, with local w→v/b and z→j/ḍž adaptation and vowel reduction qualified; the bare weight noun is not substituted for this adjective.',tier='qualified',kind='borrowed',citation='platts1884[s.v. waznī, wazanī]',source_url='https://www.rekhta.org/urdudictionary?keyword=vaznii')
for l,words in [('Malvi','bajandār|vajandār'),('Nimadi','bajanda|vajandār')]:
 b.add(l,'heavy',words,'f_47fi53dmujrya','Platts lists wazn-dār as weighty under wazn, and the existing Hindi head preserves the complete adjective. The survey forms are proposed as whole-word loans; w→v/b, z→j and the loss of final r in bajanda remain local review points, with regional mediation open.',tier='qualified',kind='borrowed',citation='platts1884[s.v. wazn, wazn-dār]',source_url='https://www.rekhta.org/urdudictionary?keyword=vazn')
b.save()
