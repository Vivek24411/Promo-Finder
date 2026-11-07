#!/usr/bin/env python3
"""Generate tests/fixtures/synthetic_show/: an ORIGINAL 10-episode microdrama ("The CEO's Hidden Wife")
written for testing and interview rehearsal. No Kuku content. Each episode uses a different format
variation so the parser is exercised the way a new show from another writing team would.

    python tests/fixtures/make_synthetic_show.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent / "synthetic_show"

# (camera description, [ (WHO, tone, text) ... ])   WHO = ACTION | SOUND FX | a character name
EPISODES = [
 (1, "At a glittering charity gala, waitress Mia is humiliated by heiress Chloe - until CEO Ethan Cole reveals Mia is his secret wife.", [
  ("Wide shot of a glittering ballroom; Mia weaves through guests with a tray of champagne.", [("ACTION", "", "Mia balances the tray, eyes down, avoiding the guests.")]),
  ("Medium shot as Chloe, in a silver gown, steps into Mia's path.", [("CHLOE", "Mocking, Cold", "Careful, waitress. This dress costs more than your whole life.")]),
  ("Snap zoom on Chloe's hand gripping a wine glass.", [("ACTION", "", "Chloe throws a glass of red wine in Mia's face."), ("SOUND FX", "", "Glass shatters on marble; the room gasps.")]),
  ("Close-up on Mia, wine dripping down her cheek, fists trembling.", [("MIA", "Trembling, Defiant", "I didn't do anything to you."), ("CHLOE", "Shouting", "Security! Throw this trash out of here!")]),
  ("Two guards grab Mia by both arms and drag her toward the doors.", [("ACTION", "", "The guards drag Mia across the floor as guests laugh.")]),
  ("Low angle: the ballroom doors burst open; Ethan Cole strides in.", [("ETHAN", "Cold, Commanding", "Take your hands off her."), ("CHLOE", "Stunned", "Ethan? She's just a waitress.")]),
  ("Push-in on Ethan as he wraps his jacket around Mia's shoulders.", [("ETHAN", "Menacing", "Touch her again and every one of you is fired.")]),
  ("Extreme close-up on Chloe's face draining of color.", [("ETHAN", "Final", "She's not a waitress. She's my wife."), ("SOUND FX", "", "A collective gasp; an ominous string sting swells into silence.")]),
 ]),
 (2, "Ethan's stepmother Vivian offers Mia a check to divorce Ethan, and threatens her brother when she refuses.", [
  ("Wide shot of a cold marble penthouse at dawn.", [("ACTION", "", "Mia steps out of the elevator in yesterday's dress.")]),
  ("Medium shot: Vivian waits at the dining table, a check in front of her.", [("VIVIAN", "Smug, Cold", "Ten million dollars. Sign the divorce papers and disappear.")]),
  ("Close-up on the check sliding across the glass table.", [("MIA", "Shaken", "Our marriage is a contract. It ends in a year anyway."), ("VIVIAN", "Condescending", "A year is too long for a girl like you.")]),
  ("Snap zoom as Mia picks up the check.", [("ACTION", "", "Mia tears up the check and lets the pieces fall."), ("MIA", "Defiant", "I don't want his money.")]),
  ("Close-up on Vivian's smile turning cruel.", [("VIVIAN", "Threatening", "Your little brother's scholarship. It would be a shame if it vanished.")]),
  ("Over-the-shoulder on Mia, frozen.", [("MIA", "Desperate", "Leave him out of this!")]),
  ("Low angle on Vivian standing, towering over Mia.", [("VIVIAN", "Chilling", "Then sign. Or you'll lose everything you love before Friday.")]),
  ("Extreme close-up on the pen in Mia's shaking hand.", [("SOUND FX", "", "A low ominous drone builds to a cliffhanger sting.")]),
 ]),
 (3, "Chloe frames Mia for leaking company secrets in front of the whole board.", [
  ("Wide shot of a glass boardroom high above the city.", [("ACTION", "", "Board members sit in silence as Chloe stands at the head of the table.")]),
  ("Medium shot on Chloe holding up a USB drive.", [("CHLOE", "Accusing", "This was found in her bag. Our launch files. Sold to our rival.")]),
  ("Close-up on Mia at the door, holding a coffee tray.", [("MIA", "Incredulous", "That's not mine. I've never seen it before.")]),
  ("Whip pan to two security guards grabbing Mia.", [("ACTION", "", "The guards seize Mia; the coffee tray crashes to the floor."), ("SOUND FX", "", "Ceramic shatters.")]),
  ("Medium shot: Ethan enters, jaw tight.", [("ETHAN", "Cold", "What is going on here?"), ("CHLOE", "Gloating", "Your wife is a spy, Ethan.")]),
  ("Close-up on the laptop screen showing a transfer receipt in Mia's name.", [("BOARD MEMBER", "Outraged", "Two hundred thousand dollars, paid to her account last night.")]),
  ("Push-in on Mia, tears in her eyes.", [("MIA", "Pleading", "Ethan, you know me. Please.")]),
  ("Extreme close-up on Ethan's unreadable face.", [("ETHAN", "Grave", "Mia. Look at me. Did you do it?"), ("SOUND FX", "", "Silence, then a sharp cliffhanger sting.")]),
 ]),
 (4, "Mia finds proof that Chloe framed her - and Chloe's car comes for her in the parking garage.", [
  ("Wide shot of an empty office at night.", [("ACTION", "", "Mia searches Chloe's desk with a phone flashlight.")]),
  ("Close-up on a phone screen lighting up in the drawer.", [("ACTION", "", "Mia reads a message: 'Transfer done. She'll never prove it.'"), ("MIA", "Whispering, Shocked", "Chloe... it was you.")]),
  ("Medium shot: Mia hurries through a dark parking garage.", [("SOUND FX", "", "Footsteps echo off the concrete.")]),
  ("Close-up on Mia dialing; the call fails.", [("MIA", "Breathless", "Come on, Ethan, pick up.")]),
  ("Wide shot: headlights flare on behind her.", [("SOUND FX", "", "An engine roars to life.")]),
  ("Snap zoom on Chloe behind the wheel, eyes cold.", [("CHLOE", "Venomous", "You should have stayed a waitress.")]),
  ("Low angle as the car speeds straight at Mia.", [("ACTION", "", "The car slams into Mia; she crashes onto the concrete."), ("SOUND FX", "", "A sickening crash, tires screaming.")]),
  ("Extreme close-up on Mia's phone on the ground, Ethan's name calling.", [("SOUND FX", "", "The phone keeps ringing; the screen cuts to black.")]),
 ]),
 (5, "In hospital, Mia learns she is pregnant - and Vivian pays the doctor to keep it hidden.", [
  ("Wide shot of a private hospital room.", [("ACTION", "", "Mia wakes slowly, bandages on her arm.")]),
  ("Medium shot: the doctor checks the chart.", [("DOCTOR", "Gentle", "You were lucky. And the baby is fine.")]),
  ("Snap zoom on Mia's face.", [("MIA", "Stunned", "The baby?"), ("DOCTOR", "Earnest", "You're eight weeks pregnant.")]),
  ("Close-up on Mia's hand resting on her stomach.", [("MIA", "Tearful", "Does Ethan know?")]),
  ("Cut to the corridor: Vivian corners the doctor.", [("VIVIAN", "Cold, Commanding", "How much does it cost for this to stay out of her file?")]),
  ("Close-up on an envelope of cash pressed into the doctor's hand.", [("DOCTOR", "Shaken", "Mrs. Cole, I can't-"), ("VIVIAN", "Menacing", "You can. Or your license disappears.")]),
  ("Push-in on Vivian looking through the window at Mia.", [("VIVIAN", "Sinister", "No one can ever know about this baby.")]),
  ("Extreme close-up on Vivian's eyes.", [("SOUND FX", "", "A deep ominous rumble into silence.")]),
 ]),
 (6, "Ethan exposes Chloe at a press conference - and Chloe threatens to reveal the contract marriage.", [
  ("Wide shot of a packed press room, cameras flashing.", [("ACTION", "", "Ethan steps up to the podium.")]),
  ("Medium shot on Ethan.", [("ETHAN", "Cold", "Last night someone tried to kill my wife.")]),
  ("Close-up on a screen behind him showing security footage of Chloe's car.", [("SOUND FX", "", "The room erupts; cameras click like rain.")]),
  ("Whip pan to Chloe at the back of the room, frozen.", [("REPORTER", "Urgent", "Miss Vance, is that your car?")]),
  ("Medium shot: Chloe pushes through the crowd toward the stage.", [("CHLOE", "Furious", "You think you've won? You've won nothing.")]),
  ("Close-up on Chloe leaning into Ethan's microphone.", [("CHLOE", "Venomous", "If I go down, I'll tell them all about your contract marriage.")]),
  ("Push-in on Ethan, jaw tight.", [("ETHAN", "Threatening", "Say one more word and I'll bury you.")]),
  ("Extreme close-up on Chloe's smile.", [("CHLOE", "Gloating", "Then I'll say it anyway."), ("SOUND FX", "", "A cliffhanger sting cuts to black.")]),
 ]),
 (7, "The contract marriage goes public; Ethan proposes to Mia for real - but she has a secret first.", [
  ("Wide shot: reporters swarm Mia outside the building.", [("REPORTER", "Mocking", "Mia! How much did he pay you to be his wife?")]),
  ("Close-up on Mia shielding her face.", [("ACTION", "", "Someone shoves Mia; she stumbles against a car.")]),
  ("Medium shot: Ethan pushes through the crowd.", [("ETHAN", "Commanding", "Get away from her. Now.")]),
  ("Wide shot as the crowd falls silent around them.", [("ACTION", "", "Ethan drops to one knee on the pavement, holding a ring.")]),
  ("Close-up on the ring box opening.", [("ETHAN", "Earnest", "No contract this time. Marry me. For real.")]),
  ("Push-in on Mia, tears in her eyes.", [("MIA", "Tearful", "Ethan, there's something you need to know first.")]),
  ("Close-up on Ethan's phone buzzing.", [("SOUND FX", "", "The phone vibrates against the pavement."), ("ETHAN", "Alarmed", "It's the hospital.")]),
  ("Extreme close-up on Mia's face going pale.", [("MIA", "Breathless", "What did they tell you?"), ("SOUND FX", "", "Silence into a cliffhanger sting.")]),
 ]),
 (8, "A quiet day at the beach house: Mia tells Ethan about the baby and they finally find peace.", [
  ("Wide shot of a sunlit beach house, waves rolling in.", [("ACTION", "", "Mia and Ethan walk barefoot along the shore.")]),
  ("Medium two-shot as they sit on the porch steps.", [("ETHAN", "Gentle", "No reporters. No board. Just us.")]),
  ("Close-up on Mia taking his hand.", [("MIA", "Soft", "I'm pregnant, Ethan.")]),
  ("Close-up on Ethan, a slow smile spreading.", [("ETHAN", "Tender", "We're having a baby?"), ("MIA", "Warm", "We are.")]),
  ("Medium shot as he pulls her into an embrace.", [("ACTION", "", "They embrace, laughing softly.")]),
  ("Close-up on their hands intertwined.", [("ETHAN", "Sincere", "Thank you for staying.")]),
  ("Wide shot of the sunset over the water.", [("ACTION", "", "They watch the sun go down, calm and content.")]),
  ("Slow push-in on the horizon.", [("SOUND FX", "", "Gentle strings fade out.")]),
 ]),
 (9, "At a charity event, the keynote speaker is Mia's mother Elena - believed dead for fifteen years.", [
  ("Wide shot of a charity auditorium.", [("ACTION", "", "Mia takes her seat beside Ethan as the lights dim.")]),
  ("Medium shot on the host at the podium.", [("HOST", "Warm", "Please welcome our founder, Elena Reyes.")]),
  ("Close-up on Mia as a woman walks on stage.", [("ACTION", "", "Mia's program slips from her hands.")]),
  ("Push-in on Mia, eyes wide.", [("MIA", "Whispering, Stunned", "Mom?")]),
  ("Wide shot: Mia stands and walks toward the stage.", [("ELENA", "Trembling", "Mia... my baby.")]),
  ("Close-up as they face each other.", [("MIA", "Sobbing", "You're alive? They told me you died in the fire.")]),
  ("Medium shot as Elena holds Mia's face.", [("ELENA", "Tearful", "I had to disappear to protect you.")]),
  ("Extreme close-up on Elena's eyes hardening.", [("ELENA", "Grave", "And Vivian Cole is the reason I did."), ("SOUND FX", "", "An ominous swell into silence.")]),
 ]),
 (10, "In the boardroom, Vivian's crimes come out - and she pulls a gun.", [
  ("Wide shot of the boardroom; Elena and Mia enter together.", [("ACTION", "", "The board members stand as Elena walks in.")]),
  ("Medium shot on Vivian at the head of the table.", [("VIVIAN", "Cold", "You should have stayed dead, Elena.")]),
  ("Close-up on Elena laying documents on the table.", [("ELENA", "Firm", "Fifteen years ago you set my house on fire. Here is the proof.")]),
  ("Whip pan to Ethan.", [("ETHAN", "Stunned", "Vivian... is this true?")]),
  ("Snap zoom on Vivian's hand pulling a gun from her bag.", [("ACTION", "", "Vivian grabs Mia and presses a gun to her side."), ("SOUND FX", "", "Chairs scrape; someone screams.")]),
  ("Close-up on Mia, frozen.", [("VIVIAN", "Screaming, Menacing", "One more step and she dies!")]),
  ("Push-in on Ethan, hands raised.", [("ETHAN", "Desperate", "Vivian, put it down. She's pregnant.")]),
  ("Extreme close-up on Vivian's finger on the trigger.", [("SOUND FX", "", "A single gunshot; the screen cuts to black.")]),
 ]),
]

# one format variation per episode (what a different writing team might send)
STYLE = {1: "kuku", 2: "no_brackets", 3: "hours", 4: "lower_dash", 5: "plain_speaker",
         6: "missing_end", 7: "one_decimal", 8: "kuku", 9: "kuku", 10: "md"}


def ts(t: float, style: str) -> str:
    m, s = divmod(t, 60)
    if style == "hours":
        return f"0:{int(m):02d}:{s:05.2f}"
    if style == "one_decimal":
        return f"{int(m):02d}:{s:04.1f}"
    return f"{int(m):02d}:{s:05.2f}"


def span(a: float, b: float | None, style: str) -> str:
    if style == "no_brackets":
        return f"Start: {ts(a, style)} | End: {ts(b, style)}"
    if style == "lower_dash":
        return f"start: {ts(a, style)} - end: {ts(b, style)}"
    if b is None:
        return f"Start: ({ts(a, style)})"
    return f"Start: ({ts(a, style)}) | End: ({ts(b, style)})"


def header(who: str, tone: str, style: str) -> str:
    label = who if who in ("ACTION", "SOUND FX", "CAMERA") else (f"{who} ({tone})" if tone else who)
    return f"{label}:" if style == "plain_speaker" else f"**{label}:**"


def render(num: int, logline: str, shots: list) -> str:
    style = STYLE[num]
    out = [f"EPISODE NUMBER: Episode {num}", f"EPISODE LOGLINE: {logline}",
           "CONTINUITY NOTES: Synthetic test episode.", "", "--- SCRIPT START ---", ""]
    t, missing_done = 0.0, False
    scene_start = 0.0
    body = []
    for cam, inner in shots:
        durs = [max(3.5, len(text) / 8) for _, _, text in inner]
        shot_len = sum(durs) + 1.6
        body += [header("CAMERA", "", style), span(t, t + shot_len, style), cam, ""]
        u = t + 0.4
        for (who, tone, text), d in zip(inner, durs):
            end = None if (style == "missing_end" and not missing_done and who not in ("ACTION", "SOUND FX")) else u + d
            missing_done = missing_done or end is None
            body += [header(who, tone, style), span(u, end, style),
                     f'"{text}"' if who not in ("ACTION", "SOUND FX") else text, ""]
            u += d
        t += shot_len
    out += ["SCENE 01", "Scene heading: INT./EXT. - VARIOUS", f"Scene start: {ts(scene_start, 'kuku')}",
            f"Scene end: {ts(t, 'kuku')}", "", "Actions and dialogue:", ""] + body + [
        header("CAMERA", "", style), span(t, t + 1.0, style), "Cut to black.", "", "--- SCRIPT END ---", ""]
    return "\n".join(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.iterdir():
        f.unlink()
    for num, logline, shots in EPISODES:
        ext = ".md" if STYLE[num] == "md" else ".txt"
        (OUT / f"CEO Hidden Wife - Ep {num}{ext}").write_text(render(num, logline, shots))
    print(f"wrote {len(EPISODES)} episodes to {OUT}")


if __name__ == "__main__":
    main()
