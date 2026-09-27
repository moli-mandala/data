"""Page-image reverse transcription pp.172 and175; source spellings preserved."""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PAGES={
(172,'left'):'''ābā	father
ādīren	arrived
ādē	known; understood
ai	now
aiang	parents
ājī	sister-in-law
ākī	axe
ālē	we
am	you
ama	your
āmbī	mango
āngluki	bathe
ānjūmē	hear
āpārange	fight
ārā	herbs
ārrākī	let go
ārūkī	make clean
ārsā	mirror
āsarū	calf
āsī	beg
āsīkī	beg
ātkom	egg
ātodin	week
awal sagīrā	beautiful
awal	good
bābā chaulī	rice
bābūlī	bābūl
badako	duck''',
(172,'right'):'''badsā	a cloud; heaven
bāgā-bāgā	slowly
bāgī,ā	servant
bakī	not
bamre	cold
bānā	bear
bāng	no; not
barē	about
barsādo	monsoon
barūn	under
batkil|bitkil	buffalo
bauñ	back
baurā	arm
bautā	bracelet; bangle
bēdai	ewe
bērī	mad
bērīā	ram; time
bhātā	bellows
bhagwān|bhagwat	God
bīānē	marry
bīde	lift; raise
bilā	kite
bin	snake
bindil	bed; bed-clothing
bing	clean
bītil	gravel
Bītwār	Thursday
bo	go''',
(175,'left'):'''kāsubā	is paining
kāt	large
kātān	grain
kātī	blacksmith
kātlānkan	fan
kātpaka	strong
kātrē	skin
kātsūm	sinner
kē,ālē	play
kērī	timber
kētkī	fasten
khabardao	be careful
kattā	sour
kattābā	is sour
khogīr	saddle
khūb	excellent
khūshī	happiness
kī	or
kiding	scorpion
kījē	sell
kira	worm
kirī	jungle-goat
kolā din	yesterday
kolā kē	take out
kolom	flour
komārai	ask
kombā	cock
kombor	body
konē	in the direction of
kor	child
korā	road
korīo	wind
kosrē	nephew; niece
kū	cough
kubrī	earthen water pot
kūhī	well
kukī	cough
kūlā	tiger
kūlafo	a lock
kulē	send
kūnjī	key''',
(175,'right'):'''kūnī	elbow
kūnkar	father-in-law
kūmū	dirty
kwāgē	beat; strike
kwālī	rabbit
lai,ē	dig
lāj	belly
lālātē	make (cakes)
landai	laugh
lang	tongue
lawa,en	tired
len	above
lo	iron
lojār	service
lojgār	wages
lokor	dry
lūbū	cloth; clothing; carpet
lūtī	native frying pan
lūtūr	ear
māchān	water-stand
māfo	forgiveness
māfokī	forgive
mahina	month
Mangrā	Tuesday
māndē	word
mānjar|mīnū	cat
māt	bamboo
maya	love
mārā	peafowl
meran	near; about
mēt	eye
mēzo	table
mī,ā	one
mī,ang	day after to-morrow
mū	nose
mū,ār	face
mūndai	kill
mūsā|mūshā	mustache
mūtkū	beads''',
}
def main():
 out=[]
 for (page,col),text in PAGES.items():
  for n,line in enumerate(text.splitlines(),1):
   form,gloss=line.split('\t')
   out.append(dict(printed_page=page,pdf_page=page+19,column=col,ordinal=n,reviewed_source_form=form,reviewed_english_gloss=gloss,review_note='Visually transcribed original page image 2026-09-26; | separates printed alternatives; Hindustani comparison labels kept in OCR evidence.'))
 assert len(out)==136
 with (ROOT/'reverse_middle_pages_review.tsv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t');w.writeheader();w.writerows(out)
if __name__=='__main__': main()
