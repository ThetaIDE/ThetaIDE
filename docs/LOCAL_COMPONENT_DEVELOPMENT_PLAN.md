# Local component development — current scope and status

The shared component-development workflow is implemented on the existing checkout layout.
This document replaces the earlier plan that allowed editing Hub originals and proposed a global
component collection. Use [Component development](../COMPONENT_DEVELOPMENT.md) for current commands.
The larger [project organization proposal](../PROJECT_ORGANIZATION_PLAN.md) remains deferred.

## Supported workflow

```text
Install from Hub → Inspect → Deep Copy model → Edit active local files → Configure → Validate → Run
```

Hub originals are read-only. Independent copying currently supports directory-based models with safely
transformable literal registrations and imports. A local copy has a new ID/version and parent provenance;
shared dependencies remain references. Unsupported kinds/layouts and identity conflicts produce errors.
Copying only YAML creates an editable profile that continues to reference shared implementation code.

The GUI's Deep Copy and Copy YAML actions and terminal component commands use shared headless services.
Editing local Python source remains an ordinary editor/terminal operation; a fresh experiment process
loads saved changes. No Apply step or publication is required.

## Current ownership and storage

| Files | Location and authority |
| --- | --- |
| Active research implementation | Existing `src/usr/` directories, with environment/plot assets at their existing paths |
| Active configurations | `in/config/`; package defaults protected, local component/profile copies editable |
| Retained package snapshots | `.thetaide/components/packages/<id>/`; used for retention/reinstall, not a synchronized working copy |
| Collection and visibility state | `.thetaide/components/collection.json` |
| Generated context | `.thetaide/components/context/<id>.md`; refreshed on Edit or `component context` |
| Hub-installed GUI plugins | `.thetaide/plugins/<id>/` by default |

Edit opens active installed source, not a retained package snapshot. On a Hub original this is inspection;
on a local copy it supports edits. Context identifies owned profiles, assets, registrations, and relevant
contracts. Keep handwritten notes separate from regenerated context.

## Shared implementation

- `src/app/workspace.py`: path policy for today's layout, including existing terminal working-directory behavior.
- `src/app/components/installer.py`, `models.py`, `dependencies.py`: headless installation and metadata;
  old frontend import paths are compatibility aliases to the same implementation/classes.
- `src/app/components/development.py`: installed-component inspection, copying and static validation.
- `src/app/components/configuration.py`: protected, atomic configuration writes and exclusive profile copies.
- `src/app/services/local_components.py`: retained collection, supported identity transforms, context and visibility.
- `src/app/services/component_exporter.py`: source/profile/asset collection and export.
- `frontend/component_actions.py`, config widgets and Hub views: desktop adapters. The desktop supplies system Trash.
- `src/app/pipeline/compose.py` and shared validation: GUI/CLI composition, including effective method overrides.

## Visibility and removal

Hide/Unhide changes local discovery preferences without moving files or invalidating existing references.
The Hub's Show hidden control keeps those entries accessible. Uninstall removes active files and retains
saved local content for reinstall; it does not leave active local source in place. Delete local component
is a separate action that shows affected locations/references and moves the owned collection/installation
to system Trash. Neither action changes remote Hub contents. Terminal component subcommands currently
cover inspection, copy, config operations, validation, context and export; they do not expose every Hub action.

## Verification and remaining boundaries

The regression tests cover independent runtime identity after copying/restart, parent preservation,
headless terminal use with GUI imports blocked, multi-profile/schema/asset preservation, conflicts,
rollback/recovery, ownership protection, configuration directives, and GUI integrations. Static validation
is not a training/shape guarantee. Filesystem read-only modes do not prevent the owning account from
changing permissions or replacing files.

Independent workspace creation/switching/migration, direct external-repository registration, a new managed
package layout, broad application-source renaming, and general copying of every component kind are deferred.
Future production releases still follow [the Hub release workflow](../THETA_HUB_RELEASE_WORKFLOW.md).
