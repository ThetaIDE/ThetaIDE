# Filetree actions

The Experiment, Components, Results, Results Files, Discussion and Text Editor filetrees use the same trash icon and right-click **Delete…** action. Delete and Backspace invoke the same action while the tree has keyboard focus. Right-click a selected row to act on the existing selection, or right-click another row to act on that row. Each tree confirms the deletion using the names and scope of its targets.

Results Files deletes individual local artifacts. Its folders organize the results listing. Use the main Results tree to delete runs or groups. In Discussion, delete a figure or select a folder to delete all its figures and their associated text, including figures hidden by the filter. The Text Editor moves selected files and folders to the system Trash and protects installed component files and their containing folders.

New filetrees inherit `FileTreeWidget` from `src.app.api.desktop`. Register each Qt tree with `_connect_file_tree`, use `_init_tree_toolbar` to build its shared toolbar, and return any other toolbar buttons from `_toolbar_groups`. Implement `deletion_targets(items)` to validate and resolve the selection, `_prompt_delete_targets(targets)` to confirm and perform the operation, and `_add_context_actions(menu, item, items)` for any additional actions. Use a stable `mode` and the host's toolbar settings store to persist button positions.
