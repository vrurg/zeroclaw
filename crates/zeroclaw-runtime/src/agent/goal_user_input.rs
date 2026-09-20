//! Transient capture of an explicit parent Goal request for user input.
//!
//! The captured request is converted into the existing durable Goal blocker
//! only after the parent turn has completed its normal tool/history pairing.
//! It intentionally does not create another control-plane store.

use std::future::Future;
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::agent::goal_child_fence::{GoalTurnScope, goal_turn_scope};
use crate::control_plane::GoalBlockerKind;

/// The durable blocker message is intentionally compact: it is displayed to
/// the user and injected into the next resumed parent turn. All Goal blocker
/// producers use this one bound because they share the same durable column.
pub(crate) const MAX_GOAL_BLOCKER_MESSAGE_CHARS: usize = 2_000;
pub(crate) const GOAL_BLOCKER_HEADING: &str = "Goal blocker";

tokio::task_local! {
    static GOAL_USER_INPUT: Arc<Mutex<Option<GoalUserInputRequest>>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GoalUserInputRequest {
    pub(crate) question: String,
    pub(crate) choices: Vec<String>,
}

/// A bounded, controller-verifiable fallback declaration from a Goal parent.
///
/// This is deliberately recognized only for a Goal parent. A child cannot use
/// the declaration to stop its parent, and ordinary agent text remains
/// unchanged.
pub(crate) struct GoalBlockerCertificate {
    pub(crate) kind: GoalBlockerKind,
    pub(crate) message: String,
}

/// Parse the exact textual fallback accepted by the Goal controller.
///
/// The parent is instructed to put this declaration at the end of its reply.
/// Providers can nevertheless append narration or a tool request afterwards.
/// Once a parent has deliberately emitted this narrow, structured signal,
/// treating later prose as a reason to keep working would defeat the request
/// and lose the user's chance to answer. Use the final valid declaration;
/// ordinary prose asking for help remains non-authoritative.
///
/// Markdown commonly places blank lines between a heading and its fields, so
/// those separators are accepted. A Markdown ATX heading may use one through
/// six `#` markers, up to three leading spaces, one or more space or tab
/// separators, and an optional closing marker run separated from the title by
/// space or tab; agent renderers routinely vary those presentation details
/// while preserving the same visible section. Any nonblank field line still
/// has to match the exact `Kind:` and `Action:` fields, so ordinary prose
/// remains non-authoritative.
pub(crate) fn candidate_goal_blocker_certificate(
    candidate: &str,
) -> Option<GoalBlockerCertificate> {
    let mut accepted = None;
    let lines = candidate.lines().collect::<Vec<_>>();
    for (heading_index, line) in lines.iter().enumerate() {
        if !is_goal_blocker_heading(line) {
            continue;
        }
        let Some((kind_index, kind_line)) =
            next_nonblank_certificate_line(&lines, heading_index + 1)
        else {
            continue;
        };
        let Some(kind) = kind_line.trim_end_matches('\r').strip_prefix("Kind: ") else {
            continue;
        };
        let Some((_, action_line)) = next_nonblank_certificate_line(&lines, kind_index + 1) else {
            continue;
        };
        let Some(message) = action_line.trim_end_matches('\r').strip_prefix("Action: ") else {
            continue;
        };
        let message = message.trim();
        if message.is_empty() || message.chars().count() > MAX_GOAL_BLOCKER_MESSAGE_CHARS {
            continue;
        }
        let kind = match kind.trim() {
            "needs_user_input" => GoalBlockerKind::NeedsUserInput,
            "human_escalation" => GoalBlockerKind::HumanEscalation,
            "external_dependency" => GoalBlockerKind::ExternalDependency,
            _ => continue,
        };
        accepted = Some(GoalBlockerCertificate {
            kind,
            message: message.to_owned(),
        });
    }
    accepted
}

/// Return the next nonblank certificate field without advancing the outer
/// heading scan. A malformed heading must not consume a later valid heading:
/// agents sometimes repeat a corrected certificate after a partial attempt.
fn next_nonblank_certificate_line<'a>(
    lines: &'a [&'a str],
    start: usize,
) -> Option<(usize, &'a str)> {
    lines
        .iter()
        .enumerate()
        .skip(start)
        .find(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| (index, *line))
}

fn is_goal_blocker_heading(line: &str) -> bool {
    let line = line.trim_end_matches('\r');
    let indentation = line.bytes().take_while(|byte| *byte == b' ').count();
    if indentation > 3 {
        return false;
    }
    let line = &line[indentation..];
    let marker_count = line.bytes().take_while(|byte| *byte == b'#').count();
    if !(1..=6).contains(&marker_count) {
        return false;
    }
    let Some(heading) = line.get(marker_count..) else {
        return false;
    };
    if !heading.starts_with([' ', '\t']) {
        return false;
    }
    let heading = heading
        .trim_start_matches([' ', '\t'])
        .trim_end_matches([' ', '\t']);
    let without_closing_markers = heading.trim_end_matches('#');
    let heading = if without_closing_markers.len() != heading.len()
        && without_closing_markers.ends_with([' ', '\t'])
    {
        without_closing_markers.trim_end_matches([' ', '\t'])
    } else {
        heading
    };
    heading == GOAL_BLOCKER_HEADING
}

/// Whether a completed model response must terminate the current parent turn
/// before the generic tool loop dispatches any requested tool calls.
///
/// The controller turns a valid certificate into the same durable pause as a
/// typed `ask_user` request. This only prevents an agent from declaring a
/// structured blocker and then executing additional work in the same response.
pub(crate) fn final_goal_blocker_ends_parent_turn(candidate: &str) -> bool {
    matches!(goal_turn_scope(), Some(GoalTurnScope::Parent))
        && candidate_goal_blocker_certificate(candidate).is_some()
}

/// Produce the one canonical user-facing form of a captured `ask_user`
/// request. Keeping both validation and persistence on this formatter avoids
/// accepting a request that cannot later be rendered durably.
pub(crate) fn format_goal_user_input_request(question: &str, choices: &[String]) -> String {
    let question = compact_span(question);
    if choices.is_empty() {
        return question;
    }
    let choices = choices
        .iter()
        .enumerate()
        .map(|(index, choice)| format!("{}. {}", index + 1, compact_span(choice)))
        .collect::<Vec<_>>()
        .join(" / ");
    format!("{question} — {choices}")
}

fn compact_span(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) enum RecordGoalUserInput {
    Recorded,
    AlreadyRequested,
}

pub(crate) async fn scope_goal_user_input<F>(future: F) -> F::Output
where
    F: Future,
{
    GOAL_USER_INPUT
        .scope(Arc::new(Mutex::new(None)), future)
        .await
}

pub(crate) async fn record_goal_user_input(
    request: GoalUserInputRequest,
) -> Option<RecordGoalUserInput> {
    let state = GOAL_USER_INPUT.try_with(Clone::clone).ok()?;
    let mut state = state.lock().await;
    if state.is_some() {
        Some(RecordGoalUserInput::AlreadyRequested)
    } else {
        *state = Some(request);
        Some(RecordGoalUserInput::Recorded)
    }
}

pub(crate) async fn take_goal_user_input() -> Option<GoalUserInputRequest> {
    let state = GOAL_USER_INPUT.try_with(Clone::clone).ok()?;
    state.lock().await.take()
}

#[cfg(test)]
mod tests {
    use super::{
        GoalUserInputRequest, RecordGoalUserInput, candidate_goal_blocker_certificate,
        final_goal_blocker_ends_parent_turn, format_goal_user_input_request,
        record_goal_user_input, scope_goal_user_input, take_goal_user_input,
    };
    use crate::agent::goal_child_fence::{scope_goal_child, scope_goal_parent};

    #[tokio::test]
    async fn parent_turn_keeps_one_exact_user_input_request() {
        scope_goal_user_input(async {
            assert!(matches!(
                record_goal_user_input(GoalUserInputRequest {
                    question: "Which option?".to_owned(),
                    choices: vec!["A".to_owned(), "B".to_owned()],
                })
                .await,
                Some(RecordGoalUserInput::Recorded)
            ));
            assert!(matches!(
                record_goal_user_input(GoalUserInputRequest {
                    question: "A second question".to_owned(),
                    choices: Vec::new(),
                })
                .await,
                Some(RecordGoalUserInput::AlreadyRequested)
            ));
            assert_eq!(
                take_goal_user_input().await,
                Some(GoalUserInputRequest {
                    question: "Which option?".to_owned(),
                    choices: vec!["A".to_owned(), "B".to_owned()],
                })
            );
        })
        .await;
    }

    #[test]
    fn formatter_is_the_canonical_compact_projection() {
        assert_eq!(
            format_goal_user_input_request(
                " Which\n policy  should I use? ",
                &[" Keep   A ".to_owned(), "Switch to B".to_owned()],
            ),
            "Which policy should I use? — 1. Keep A / 2. Switch to B"
        );
    }

    #[tokio::test]
    async fn structured_certificate_ends_only_a_goal_parent_turn() {
        let candidate =
            "Need a decision.\n## Goal blocker\nKind: needs_user_input\nAction: Choose A or B";
        assert!(candidate_goal_blocker_certificate(candidate).is_some());
        assert!(!final_goal_blocker_ends_parent_turn(candidate));

        scope_goal_parent(async {
            assert!(final_goal_blocker_ends_parent_turn(candidate));
            scope_goal_child(Box::pin(async {
                assert!(!final_goal_blocker_ends_parent_turn(candidate));
            }))
            .await;
        })
        .await;

        let continued =
            "## Goal blocker\nKind: needs_user_input\nAction: Choose A\nMore work follows";
        assert!(candidate_goal_blocker_certificate(continued).is_some());
        scope_goal_parent(async {
            assert!(final_goal_blocker_ends_parent_turn(continued));
        })
        .await;

        let markdown_spaced = "## Goal blocker\n\nKind: needs_user_input\n\nAction: Choose A or B";
        let parsed = candidate_goal_blocker_certificate(markdown_spaced).unwrap();
        assert_eq!(
            parsed.kind,
            crate::control_plane::GoalBlockerKind::NeedsUserInput
        );
        assert_eq!(parsed.message, "Choose A or B");
        scope_goal_parent(async {
            assert!(final_goal_blocker_ends_parent_turn(markdown_spaced));
        })
        .await;

        let alternate_heading = "# Goal blocker\nKind: needs_user_input\nAction: Choose A or B";
        assert!(candidate_goal_blocker_certificate(alternate_heading).is_some());
        scope_goal_parent(async {
            assert!(final_goal_blocker_ends_parent_turn(alternate_heading));
        })
        .await;

        for standard_heading in [
            "   ###  Goal blocker\t  ###",
            "#\tGoal blocker",
            "###### Goal blocker   ######",
        ] {
            let candidate =
                format!("{standard_heading}\nKind: needs_user_input\nAction: Choose A or B");
            assert!(candidate_goal_blocker_certificate(&candidate).is_some());
            scope_goal_parent(async {
                assert!(final_goal_blocker_ends_parent_turn(&candidate));
            })
            .await;
        }

        for invalid_heading in [
            "## Goal blocker checklist",
            "##Goal blocker",
            "## Goal blocker#",
            "####### Goal blocker",
            "## Goal blocker\u{00A0}",
        ] {
            assert!(
                candidate_goal_blocker_certificate(&format!(
                    "{invalid_heading}\nKind: needs_user_input\nAction: Choose A or B"
                ))
                .is_none(),
                "{invalid_heading:?} must not be accepted as a Goal blocker heading"
            );
        }
        let corrected_after_partial =
            "## Goal blocker\n\n## Goal blocker\nKind: needs_user_input\nAction: Choose A or B";
        assert!(
            candidate_goal_blocker_certificate(corrected_after_partial).is_some(),
            "a partial certificate must not consume a later corrected certificate"
        );
        scope_goal_parent(async {
            assert!(final_goal_blocker_ends_parent_turn(corrected_after_partial));
        })
        .await;
        assert!(
            candidate_goal_blocker_certificate(
                "    ## Goal blocker\nKind: needs_user_input\nAction: Choose A or B"
            )
            .is_none(),
            "four spaces are a Markdown code block, not a heading"
        );

        assert!(
            candidate_goal_blocker_certificate("I need a decision before I can continue.")
                .is_none()
        );
    }
}
