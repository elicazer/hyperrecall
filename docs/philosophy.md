# Philosophy

*Why a hypergraph, why neuroscience, and what MeshMind is really for.*

## Memory is not storage

The AI industry treats memory as a storage problem: where do we put the text so
we can fetch it later? Databases, vector indexes, retrieval pipelines — all
answering "how do we *store and find* strings."

But human memory isn't a filing cabinet you retrieve from. It's a living
associative structure that *reconstructs*. You don't look up "my tenth birthday";
a cue — a smell, a song — activates a node, activation spreads, and a whole scene
reassembles, complete with the people who were there and how you felt. Memories
fade when unused and sharpen when revisited. New information doesn't overwrite old
information; it sits alongside it, sometimes in tension.

MeshMind takes that seriously as an engineering spec, not a metaphor.

## Why hypergraph

The smallest honest unit of experience is not a pair. "Eli asked David about
TEDx on July 13" is not `Eli → David`; it's a single event binding a person, a
person, a topic, a time — *at once*. The moment you force that into binary
`(head, relation, tail)` triples, you invent hub nodes to hold the pieces
together and you lose the fact that they were ever one thing. That loss is
invisible until you try to recall the episode and get back disconnected shards.

A hyperedge keeps the episode whole. Arity is arbitrary; each participant keeps
its role. This is the one non-negotiable in MeshMind: **if it ever became a
triple store, it would have deleted its own reason to exist.** A test enforces
it.

## Why neuroscience-inspired

Four borrowed mechanisms, each earning its place:

- **Spreading activation** gives retrieval *structure*. You don't get a bag of
  nearby chunks; you get a connected subgraph — memory with its relationships
  attached, which is what a reasoning agent actually needs.
- **Decay** gives memory *time*. What isn't used fades, so the working set stays
  relevant instead of growing monotonically into noise.
- **Reinforcement** gives memory *salience*. What's recalled often gets stronger,
  a feedback loop that surfaces what matters.
- **Contradiction and supersession** give memory *epistemics*. Beliefs conflict;
  beliefs change. A memory system that silently overwrites is lying about its own
  history. MeshMind keeps both sides, flags them, and lets the agent reason.

None of this requires a neural network. It's a handful of well-chosen dynamics
over a graph — cheap, inspectable, and honest.

## Why portable, why open

Your agent's memory is among the most personal artifacts software will ever hold
— what it knows about you, your projects, your decisions. It should not be
trapped inside one vendor's proprietary store. MeshMind exports to a directory of
plain Markdown files you can read, diff, edit, back up, and carry between tools:
Claude, Cursor, OpenClaw, your own scripts. "USB-C for AI memory."

And it's Apache 2.0, forever. The substrate for how AI agents remember is
infrastructure. Infrastructure this important should be open, forkable, and owned
by no one.

## The pitch, in one breath

> Current AI memory forgets what compaction deletes, flattens what vector stores
> can't structure, and shreds what knowledge graphs can't hold whole. MeshMind
> stores memory the way brains do — a hypergraph of episodes, retrieved by
> spreading activation, that decays, strengthens, contradicts, and supersedes —
> in a portable, open format that belongs to you.
